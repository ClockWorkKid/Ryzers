# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""Non-destructive runtime optimizations for the in-process X-WAM runner.

These are the latency reductions validated by the agent latency sweep
(agent_scripts/latency_opt/xwam_opt_sweep2.py): they cut per-step / per-region
compute WITHOUT changing the number of denoise steps, so the policy still runs
the full 10 action + 50 video steps. Applied as reversible monkeypatches on a
live `runners.xwam_runner.XWAMRunner` (no upstream edits, rule 2.1).

Levers (each toggleable; all default-on once XWAM_OPT=1):
  * vae_bf16      : run the Wan2.2 VAE conv encoder in bf16 instead of fp32.
                    Upstream forces fp32 (Wan2_2_VAE.dtype=torch.float, encode()
                    wraps in amp.autocast(dtype=self.dtype)) despite bf16 weights,
                    so the ~4s MIOpen conv never touches the bf16 matrix cores.
                    Single biggest precision lever (~3.3x on VAE encode).
  * bf16_all      : globally rewrite torch.amp.autocast(dtype=float32) -> bf16, so
                    the DiT modulation, per-token timestep embed+proj, and the
                    action/proprio encoders/decoders run on the bf16 cores too.
                    NOTE: the DiT asserts `e.dtype == float32` inside every block;
                    run the interpreter with `python -O` so those asserts are
                    stripped (this module warns if bf16 is on but asserts live).
  * context_cache : memoize the text-embedding projection + per-block cross-attn
                    K/V (the text condition is constant across denoise steps).
                    Bit-exact. Cleared at the start of every generate() so a new
                    observation/instruction never reuses a stale K/V.
  * text_cache    : memoize the UMT5 text encode by prompt string. Bit-exact; the
                    instruction is constant across an episode, so this persists
                    across generate() calls (keyed by prompt, never by pointer).
  * freqs_cache   : memoize RoPE freqs across steps (bit-exact). Off by default
                    (excluded from the validated stack; negligible in-model gain).

Selected by env var XWAM_OPT=1 via the DirectXWAM constructor hook. Default-off,
so an unset XWAM_OPT leaves the runner bit-exact-identical to upstream.
"""
import os

import torch


def _flag(name, default):
    v = os.environ.get(name)
    if v is None:
        return default
    return v.strip().lower() in {"1", "true", "yes", "on"}


def _patch_vae_bf16(runner):
    runner.vae.dtype = torch.bfloat16
    return lambda: None


def _patch_bf16_all():
    """Rewrite every torch.amp.autocast(dtype=float32) context to bf16.
    Decorator-applied autocasts evaluated at import (e.g. RoPE) are unaffected."""
    import torch.amp as amp
    orig = amp.autocast

    def ac(*args, **kwargs):
        if kwargs.get("dtype", None) is torch.float32:
            kwargs["dtype"] = torch.bfloat16
        elif len(args) >= 2 and args[1] is torch.float32:
            args = (args[0], torch.bfloat16) + tuple(args[2:])
        return orig(*args, **kwargs)

    amp.autocast = ac
    return lambda: setattr(amp, "autocast", orig)


def _patch_context_cache(runner, caches):
    """text-embed proj + per-block cross-attn K/V cache (bit-exact within a
    generate). Registers its caches in `caches` so generate() can clear them."""
    from modules.attention import attention

    te = runner.model.text_embedding
    te_orig = te.forward
    te_cache = {}
    caches.append(te_cache)

    def te_fwd(x):
        k = x.data_ptr()
        if k not in te_cache:
            te_cache.clear()
            te_cache[k] = te_orig(x)
        return te_cache[k]
    te.forward = te_fwd

    for blk in runner.model.blocks:
        ca = blk.cross_attn
        cache = {}
        caches.append(cache)

        def make(ca=ca, cache=cache):
            def fwd(x, context):
                b, n, d = x.size(0), ca.num_heads, ca.head_dim
                x = x.type_as(ca.q.weight)
                q = ca.norm_q(ca.q(x)).view(b, -1, n, d)
                key = context.data_ptr()
                if key not in cache:
                    cache.clear()
                    cache[key] = (ca.norm_k(ca.k(context)).view(b, -1, n, d),
                                  ca.v(context).view(b, -1, n, d))
                k, v = cache[key]
                return ca.o(attention(q, k, v).flatten(2))
            return fwd
        ca.forward = make()
    return lambda: None


def _patch_text_cache(runner):
    """Memoize UMT5 encode by prompt string (bit-exact; persists across steps)."""
    te = getattr(runner, "text_encoder", None)
    orig = getattr(te, "forward", None)
    if orig is None:
        return lambda: None
    cache = {}

    def fwd(prompts, *a, **k):
        key = tuple(prompts) if isinstance(prompts, (list, tuple)) else prompts
        if key not in cache:
            cache[key] = orig(prompts, *a, **k)
        return cache[key]
    te.forward = fwd
    return lambda: None


def _patch_freqs_cache(runner, caches):
    m = runner.model
    orig = m._create_freqs
    cache = {}
    caches.append(cache)

    def fn(grid_size, start_frame=0):
        key = (tuple(int(x) for x in grid_size), int(start_frame))
        if key not in cache:
            cache[key] = orig(grid_size, start_frame)
        return cache[key]
    m._create_freqs = fn
    return lambda: None


def _kv_prefill_forward(model, x, t, context, actions, t_actions, proprios, t_proprios):
    """Full DiT forward (run_depth=False, no CFG) that ALSO caches per-layer VIDEO-token
    K/V into ``model._kv_cache`` (and the constant text ctx/freqs into ``model._kv_meta``).
    Faithful copy of ``XWAMModel._forward_single``'s run_depth=False path + a capture hook."""
    from einops import rearrange
    from modules.wan_model import sinusoidal_embedding_1d

    T = x.shape[2]
    Ta = actions.shape[1]
    Tp = proprios.shape[1]

    x = x.type_as(model.patch_embedding.weight)
    x = model.patch_embedding(rearrange(x, "b c t v h w -> (b v) c t h w"))
    grid_sizes = list(x.shape[2:])
    video_freqs = model._create_freqs(grid_sizes)
    x = x.flatten(2).transpose(1, 2)
    x = rearrange(x, "(b v) l d-> b v l d", v=model.num_views)
    with torch.amp.autocast("cuda", dtype=torch.float32):
        view_ids = torch.arange(model.num_views, device=x.device)
        view_embeddings = model.view_embedding(view_ids).view(1, model.num_views, 1, -1)
        x = x + view_embeddings
    x = rearrange(x, "b v (t hw) d-> b (t v hw) d", v=model.num_views, t=T)
    with torch.amp.autocast("cuda", dtype=torch.float32):
        actions_e = model.action_encoder(actions)
        proprios_e = model.proprio_encoder(proprios)

    n_video = x.shape[1]
    input_seq = torch.cat([x, actions_e, proprios_e], dim=1)
    freqs = torch.cat([video_freqs, model.action_freqs[:Ta], model.proprio_freqs[:Tp]], dim=0)
    context_e = model.text_embedding(context)

    if t.ndim == 1:
        t = t.view(t.size(0), 1).repeat(1, T)
    t = t.repeat_interleave(n_video // T, dim=1)
    if t_actions.ndim == 1:
        t_actions = t_actions.view(t_actions.size(0), 1).repeat(1, Ta)
    if t_proprios.ndim == 1:
        t_proprios = t_proprios.view(t_proprios.size(0), 1).repeat(1, Tp)
    with torch.amp.autocast("cuda", dtype=torch.float32):
        t_seq = torch.cat([t, t_actions, t_proprios], dim=1)
        e = model.time_embedding(
            sinusoidal_embedding_1d(model.freq_dim, t_seq.flatten()).unflatten(0, t_seq.shape).float()
        )
        e0 = model.time_projection(e).unflatten(2, (6, model.dim))

    model._kv_cache = []
    for bi in range(model.num_layers):
        input_seq, rope_k, v = model.blocks[bi](
            input_seq, e=e0, freqs=freqs, context=context_e, save_kv_cache=True
        )
        model._kv_cache.append((rope_k[:, :n_video].detach(), v[:, :n_video].detach()))
    model._kv_meta = {"context_e": context_e, "grid_sizes": grid_sizes, "T": T, "n_video": n_video}

    xv, av, pv = input_seq.split([n_video, Ta, Tp], dim=1)
    xv = model.head(xv, e[:, :n_video])
    with torch.amp.autocast("cuda", dtype=torch.float32):
        av = model.action_decoder(av)
        pv = model.proprio_decoder(pv)
    xv = rearrange(xv, "b (t v hw) c -> (b v) (t hw) c", t=T, v=model.num_views)
    xv = model.unpatchify(xv, grid_sizes)
    xv = rearrange(xv, "(b v) c t h w -> b c t v h w", v=model.num_views)
    return xv.float(), av.float(), pv.float()


def _kv_reduced_forward(model, actions, t_actions, proprios, t_proprios):
    """Reduced DiT forward over ONLY the action+proprio tokens (queries), attending to the
    frozen per-layer video K/V from the last prefill. Skips ~99% of the per-step token GEMMs
    (video Q/K/V/FFN/out-proj are never recomputed). Returns (vt_actions, vt_proprios)."""
    from modules.wan_model import sinusoidal_embedding_1d

    Ta = actions.shape[1]
    Tp = proprios.shape[1]
    context_e = model._kv_meta["context_e"]
    with torch.amp.autocast("cuda", dtype=torch.float32):
        actions_e = model.action_encoder(actions)
        proprios_e = model.proprio_encoder(proprios)
    input_seq = torch.cat([actions_e, proprios_e], dim=1)
    freqs = torch.cat([model.action_freqs[:Ta], model.proprio_freqs[:Tp]], dim=0)

    if t_actions.ndim == 1:
        t_actions = t_actions.view(t_actions.size(0), 1).repeat(1, Ta)
    if t_proprios.ndim == 1:
        t_proprios = t_proprios.view(t_proprios.size(0), 1).repeat(1, Tp)
    with torch.amp.autocast("cuda", dtype=torch.float32):
        t_seq = torch.cat([t_actions, t_proprios], dim=1)
        e = model.time_embedding(
            sinusoidal_embedding_1d(model.freq_dim, t_seq.flatten()).unflatten(0, t_seq.shape).float()
        )
        e0 = model.time_projection(e).unflatten(2, (6, model.dim))

    for bi in range(model.num_layers):
        ck, cv = model._kv_cache[bi]
        input_seq = model.blocks[bi](input_seq, e=e0, freqs=freqs, context=context_e, cache_k=ck, cache_v=cv)

    av, pv = input_seq.split([Ta, Tp], dim=1)
    with torch.amp.autocast("cuda", dtype=torch.float32):
        av = model.action_decoder(av)
        pv = model.proprio_decoder(pv)
    return av.float(), pv.float()


def _patch_kv_prefill(runner, prefill_steps, log=True):
    """Replace runner.forward with a video-prefill / cross-step K/V-reuse variant for the
    deployment action path (early_stop, cfg=0, run_depth=False). First ``prefill_steps`` (K)
    denoise steps are normal full joint forwards (video keeps denoising, scheduler stays
    consecutive) that cache per-layer video K/V; the remaining steps freeze the video and run
    the cheap action+proprio-only reduced forward. Any other path falls back to upstream."""
    import torch as _torch
    from utils.fm_solvers_unipc import FlowUniPCMultistepScheduler

    orig_forward = runner.forward
    cfg_obj = runner.config

    def forward_kv(batch, seeds=None, early_stop=False, cfg=0.0):
        # Only the deployment action path is eligible; everything else uses upstream exactly.
        if (not early_stop) or cfg > 0.0 or getattr(runner, "run_depth", False):
            return orig_forward(batch, seeds=seeds, early_stop=early_stop, cfg=cfg)

        context_embeddings, gt_latents, _ = runner._prepare_condition(batch)
        B, C, T, MV, H, W = gt_latents.shape
        gt_actions = batch["actions"].float()
        gt_proprios = batch["proprios"].float()
        context_for_model = context_embeddings

        if seeds is not None:
            nl, na, npr = [], [], []
            for i, seed in enumerate(seeds):
                gen = _torch.Generator(device=runner.device).manual_seed(int(seed))
                nl.append(_torch.randn(gt_latents[i:i + 1].shape, generator=gen, device=runner.device, dtype=gt_latents.dtype))
                na.append(_torch.randn(gt_actions[i:i + 1].shape, generator=gen, device=runner.device, dtype=gt_actions.dtype))
                npr.append(_torch.randn(gt_proprios[i:i + 1].shape, generator=gen, device=runner.device, dtype=gt_proprios.dtype))
            noise_latents, noise_actions, noise_proprios = _torch.cat(nl), _torch.cat(na), _torch.cat(npr)
        else:
            noise_latents = _torch.randn(gt_latents.shape, generator=runner.generator_per_rank, dtype=gt_latents.dtype).to(runner.device)
            noise_actions = _torch.randn(gt_actions.shape, generator=runner.generator_per_rank, dtype=gt_actions.dtype).to(runner.device)
            noise_proprios = _torch.randn(gt_proprios.shape, generator=runner.generator_per_rank, dtype=gt_proprios.dtype).to(runner.device)

        latent_mask = _torch.zeros((B, 1, T, 1, 1, 1), dtype=_torch.long, device=runner.device)
        latent_mask[:, :, 0] = 1
        xt_latents = gt_latents * latent_mask + noise_latents * (1 - latent_mask)
        action_mask = _torch.zeros((B, gt_actions.shape[1], 1), dtype=_torch.long, device=runner.device)
        xt_actions = gt_actions * action_mask + noise_actions * (1 - action_mask)
        proprio_mask = _torch.zeros((B, gt_proprios.shape[1], 1), dtype=_torch.long, device=runner.device)
        proprio_mask[:, 0] = 1
        xt_proprios = gt_proprios * proprio_mask + noise_proprios * (1 - proprio_mask)

        ads = cfg_obj.action_denoise_steps if cfg_obj.use_decoupled_inference else cfg_obj.sample_steps

        def _sched(n):
            s = FlowUniPCMultistepScheduler(num_train_timesteps=cfg_obj.flow_matching_num_train_timesteps,
                                            shift=1, use_dynamic_shifting=False)
            s.set_timesteps(n, device=runner.device, shift=cfg_obj.time_shifting)
            return s
        sch_v = _sched(cfg_obj.sample_steps)
        sch_a = _sched(ads)
        sch_p = _sched(ads)
        v_ts, a_ts, p_ts = sch_v.timesteps, sch_a.timesteps, sch_p.timesteps

        K = max(1, int(prefill_steps))
        model = runner.model
        for ti in range(cfg_obj.sample_steps):
            video_t = v_ts[ti]
            if cfg_obj.use_decoupled_inference and ti >= ads:
                break  # early_stop
            action_t, proprio_t = a_ts[ti], p_ts[ti]
            latent_ts = video_t * (1 - latent_mask).view(B, T)
            action_ts = action_t * (1 - action_mask).view(B, gt_actions.shape[1])
            proprio_ts = proprio_t * (1 - proprio_mask).view(B, gt_proprios.shape[1])

            if ti < K:
                vt_lat, vt_act, vt_pro = _kv_prefill_forward(
                    model, xt_latents, latent_ts, context_for_model, xt_actions, action_ts, xt_proprios, proprio_ts)
                xt_latents = sch_v.step(vt_lat, video_t, xt_latents, return_dict=False)[0]
                xt_latents = gt_latents * latent_mask + xt_latents * (1 - latent_mask)
            else:
                vt_act, vt_pro = _kv_reduced_forward(model, xt_actions, action_ts, xt_proprios, proprio_ts)

            xt_actions = sch_a.step(vt_act, action_t, xt_actions, return_dict=False)[0]
            xt_actions = gt_actions * action_mask + xt_actions * (1 - action_mask)
            xt_proprios = sch_p.step(vt_pro, proprio_t, xt_proprios, return_dict=False)[0]
            xt_proprios = gt_proprios * proprio_mask + xt_proprios * (1 - proprio_mask)

        return xt_latents, xt_actions, xt_proprios, None

    runner.forward = forward_kv
    if log:
        print(f"[xwam_opts] KV-prefill enabled: prefill_steps(K)={prefill_steps} "
              f"(full joint steps, then frozen-video action-only reduced forward)", flush=True)
    return lambda: setattr(runner, "forward", orig_forward)


def apply_opts(runner, log=True):
    """Apply the enabled non-destructive optimizations in place on `runner`.

    Returns the list of enabled optimization names. Reads per-lever env toggles
    (all default-on once this is called). The caller is responsible for gating on
    XWAM_OPT=1 so an unset env leaves the runner untouched.
    """
    enabled = []
    caches = []  # per-generate-cleared caches (context cache)

    if _flag("XWAM_OPT_VAE_BF16", True):
        _patch_vae_bf16(runner)
        enabled.append("vae_bf16")

    want_bf16 = _flag("XWAM_OPT_BF16", True)
    if want_bf16:
        _patch_bf16_all()
        enabled.append("bf16_all")
        if __debug__ and log:
            print("[xwam_opts] WARNING: bf16_all is on but interpreter asserts are "
                  "live (__debug__=True). The DiT asserts e.dtype==float32 and will "
                  "raise. Run with `python -O` to strip them.", flush=True)

    if _flag("XWAM_OPT_CONTEXT_CACHE", True):
        _patch_context_cache(runner, caches)
        enabled.append("context_cache")

    if _flag("XWAM_OPT_TEXT_CACHE", True):
        _patch_text_cache(runner)
        enabled.append("text_cache")

    if _flag("XWAM_OPT_FREQS_CACHE", False):
        _patch_freqs_cache(runner, caches)
        enabled.append("freqs_cache")

    # Structural (approximate) lever: video-prefill + cross-step K/V reuse. OFF by default
    # (not bit-exact — freezes the video branch after K joint steps), gated by XWAM_KV=1.
    if _flag("XWAM_KV", False):
        k = os.environ.get("XWAM_KV_PREFILL_STEPS")
        _patch_kv_prefill(runner, int(k) if k else 3, log=log)
        enabled.append("kv_prefill")

    # Clear the per-generate caches (context K/V) at the start of every generate so
    # a fresh observation/instruction never reuses a stale, pointer-keyed entry.
    if caches:
        gen_orig = runner.generate

        def generate(*a, **k):
            for c in caches:
                c.clear()
            return gen_orig(*a, **k)
        runner.generate = generate

    if log:
        print(f"[xwam_opts] applied non-destructive opts: {enabled}", flush=True)
    return enabled
