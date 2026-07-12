"""Distillation trainer: MolmoAct2 seam -> tiny student, via torchdistill.

Thin fork of torchdistill's image_classification example. torchdistill owns the
loop, DDP, forward hooks, scheduling and checkpointing; we only swap the
classification-accuracy eval for a seam feature-fidelity eval (cosine / rel-L2),
since this is a label-free feature-regression distillation.

    torchrun --nproc_per_node=N train.py --config configs/distill_hybrid.yaml \
        --run_log logs/run.log

Importing `distill.register` populates torchdistill's registries with the
MolmoAct2 teacher, the student, and the seam loss referenced by the YAML.
"""

from __future__ import annotations

import argparse
import datetime
import os
import time

import torch
from torch import distributed as dist
from torch.backends import cudnn

from torchdistill.common import file_util, module_util, yaml_util
from torchdistill.common.constant import def_logger
from torchdistill.common.main_util import (
    import_dependencies,
    init_distributed_mode,
    is_main_process,
    load_ckpt,
    save_ckpt,
    set_seed,
)
from torchdistill.core.distillation import get_distillation_box
from torchdistill.datasets.util import build_data_loader
from torchdistill.misc.log import MetricLogger, SmoothedValue, set_basic_log_config, setup_log_file
from torchdistill.models.registry import get_model

import distill.register  # noqa: F401  (registers teacher/student/loss)
from distill.losses import fidelity_metrics

logger = def_logger.getChild(__name__)


def get_args():
    p = argparse.ArgumentParser(description="MolmoAct2 vision-encoder distillation")
    p.add_argument("--config", required=True, help="yaml config path")
    p.add_argument("--device", default="cuda")
    p.add_argument("--run_log", help="log file path")
    p.add_argument("--start_epoch", default=0, type=int)
    p.add_argument("--seed", type=int)
    p.add_argument("-disable_cudnn_benchmark", action="store_true")
    p.add_argument("-test_only", action="store_true")
    p.add_argument("--world_size", default=1, type=int)
    p.add_argument("--dist_url", default="env://")
    p.add_argument("-adjust_lr", action="store_true")
    return p.parse_args()


def load_model(model_config, device):
    model = get_model(model_config["key"], model_config.get("repo_or_dir"), **model_config.get("kwargs", {}))
    load_ckpt(model_config.get("src_ckpt"), model=model, strict=False)
    return model.to(device)


def train_one_epoch(training_box, device, epoch, log_freq):
    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", SmoothedValue(window_size=1, fmt="{value}"))
    header = f"Epoch: [{epoch}]"
    for sample_batch, targets, supp_dict in metric_logger.log_every(
        training_box.train_data_loader, log_freq, header
    ):
        sample_batch = sample_batch.to(device)
        if torch.is_tensor(targets):
            targets = targets.to(device)
        loss = training_box.forward_process(sample_batch, targets, supp_dict)
        training_box.post_forward_process(loss=loss)
        metric_logger.update(loss=loss.item(), lr=training_box.optimizer.param_groups[0]["lr"])
        if (torch.isnan(loss) or torch.isinf(loss)) and is_main_process():
            raise ValueError(f"Training diverged: loss={loss}")


@torch.inference_mode()
def evaluate_fidelity(student_model, teacher_model, data_loader, device, header="Val:"):
    student_model.eval()
    teacher_model.eval()
    metric_logger = MetricLogger(delimiter="  ")
    for sample_batch, _ in metric_logger.log_every(data_loader, 100, header):
        sample_batch = sample_batch.to(device)
        s = student_model(sample_batch)
        t = teacher_model(sample_batch)
        m = fidelity_metrics(s, t)
        metric_logger.meters["cosine"].update(m["cosine"], n=sample_batch.shape[0])
        metric_logger.meters["rel_l2"].update(m["rel_l2"], n=sample_batch.shape[0])
    metric_logger.synchronize_between_processes()
    cosine = metric_logger.cosine.global_avg
    rel_l2 = metric_logger.rel_l2.global_avg
    logger.info(f" * seam cosine {cosine:.5f}  rel_l2 {rel_l2:.5f}")
    return cosine


def train(teacher_model, student_model, dataset_dict, src_ckpt, dst_ckpt,
          device, device_ids, distributed, world_size, config, args):
    train_config = config["train"]
    lr_factor = world_size if distributed and args.adjust_lr else 1
    box = get_distillation_box(
        teacher_model, student_model, dataset_dict, train_config, device, device_ids, distributed, lr_factor
    )
    best = -1.0
    optimizer, lr_scheduler = box.optimizer, box.lr_scheduler
    if file_util.check_if_exists(src_ckpt):
        best, _ = load_ckpt(src_ckpt, optimizer=optimizer, lr_scheduler=lr_scheduler)
    log_freq = train_config["log_freq"]
    wrapped_student = box.student_model
    t0 = time.time()
    for epoch in range(args.start_epoch, box.num_epochs):
        box.pre_epoch_process(epoch=epoch)
        train_one_epoch(box, device, epoch, log_freq)
        cosine = evaluate_fidelity(wrapped_student, teacher_model, box.val_data_loader, device, header="Val:")
        if cosine > best:
            best = cosine
            if is_main_process():
                logger.info(f"New best seam cosine {best:.5f} -> saving {dst_ckpt}")
            save_ckpt(wrapped_student, optimizer, lr_scheduler, best, args, dst_ckpt)
        box.post_epoch_process()
    if distributed:
        dist.barrier()
    logger.info("Training time " + str(datetime.timedelta(seconds=int(time.time() - t0))))
    box.clean_modules()


def main(args):
    set_basic_log_config()
    if is_main_process() and args.run_log is not None:
        setup_log_file(os.path.expanduser(args.run_log))
    distributed, world_size, device_ids = init_distributed_mode(args.world_size, args.dist_url)
    logger.info(args)
    if not args.disable_cudnn_benchmark:
        cudnn.benchmark = True
    set_seed(args.seed)
    config = yaml_util.load_yaml_file(os.path.expanduser(args.config))
    import_dependencies(config.get("dependencies", None))
    device = torch.device(args.device)
    dataset_dict = config["datasets"]
    models_config = config["models"]
    teacher_model = load_model(models_config["teacher_model"], device)
    student_config = models_config["student_model"]
    src_ckpt = student_config.get("src_ckpt")
    dst_ckpt = student_config["dst_ckpt"]
    student_model = load_model(student_config, device)

    if not args.test_only:
        train(teacher_model, student_model, dataset_dict, src_ckpt, dst_ckpt,
              device, device_ids, distributed, world_size, config, args)

    student_wo = student_model.module if module_util.check_if_wrapped(student_model) else student_model
    load_ckpt(dst_ckpt, model=student_wo, strict=True)
    test_config = config["test"]
    tdl_cfg = test_config["test_data_loader"]
    test_loader = build_data_loader(dataset_dict[tdl_cfg["dataset_id"]], tdl_cfg, distributed)
    evaluate_fidelity(student_wo, teacher_model, test_loader, device, header="Test:")


if __name__ == "__main__":
    main(get_args())
