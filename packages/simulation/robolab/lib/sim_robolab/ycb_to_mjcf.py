# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""Fetch YCB google_16k meshes and convert them to robosuite-loadable MJCF objects.

The real RoboLab `banana_bowl.usda` scene uses the YCB `011_banana` and `024_bowl` meshes.
This module fetches those SAME source meshes from the open YCB benchmark S3 bucket at
RUNTIME (rules 3 & 8 -- never vendor/bake third-party binaries) and converts each into a
robosuite `MujocoXMLObject`-compatible MJCF:

  - visual  : the full textured OBJ (group 1, no collision), with its texture_map.png.
  - collision: a CONVEX DECOMPOSITION (coacd) of the mesh (group 0). Critical for the bowl:
               a single convex hull would fill the cavity and defeat containment.
  - sites    : bottom_site / top_site / horizontal_radius_site from the mesh bounds
               (required by robosuite's placement samplers).

Also writes `<name>_meta.json` with the container geometry (inner radius, base/rim offsets)
so the success predicate can read faithful numbers.

YCB meshes are already in metres and in a canonical upright frame (bowl opening +z), so no
rescaling is needed. Units: metres.
"""
import json
import os
import shutil
import tarfile
import urllib.request

import numpy as np
import trimesh

YCB_BASE = "http://ycb-benchmarks.s3-website-us-east-1.amazonaws.com/data/google"

# RoboLab banana_bowl.usda -> YCB objects (see assets/objects/object_catalog.json upstream).
YCB_OBJECTS = {
    "banana": {"ycb_id": "011_banana", "friction": (2.0, 0.3, 0.1), "density": 250.0,
               "is_container": False},
    "bowl": {"ycb_id": "024_bowl", "friction": (1.0, 0.3, 0.1), "density": 400.0,
             "is_container": True},
}


def _download_extract(ycb_id, cache_dir):
    obj_dir = os.path.join(cache_dir, ycb_id)
    mesh_obj = os.path.join(obj_dir, "google_16k", "textured.obj")
    if os.path.exists(mesh_obj):
        return mesh_obj
    os.makedirs(cache_dir, exist_ok=True)
    tgz = os.path.join(cache_dir, f"{ycb_id}_google_16k.tgz")
    url = f"{YCB_BASE}/{ycb_id}_google_16k.tgz"
    print(f"[ycb] downloading {url}", flush=True)
    urllib.request.urlretrieve(url, tgz)
    with tarfile.open(tgz) as t:
        t.extractall(cache_dir)
    os.remove(tgz)
    if not os.path.exists(mesh_obj):
        raise FileNotFoundError(f"expected {mesh_obj} after extracting {ycb_id}")
    return mesh_obj


def _coacd_decompose(mesh, max_parts_hint=24):
    """Convex-decompose a trimesh.Trimesh into a list of convex trimesh parts."""
    import coacd

    cmesh = coacd.Mesh(mesh.vertices, mesh.faces)
    parts = coacd.run_coacd(cmesh, threshold=0.05)
    out = []
    for verts, faces in parts:
        out.append(trimesh.Trimesh(vertices=np.asarray(verts), faces=np.asarray(faces)))
    return out


def convert(name, spec, assets_root, texture_src_dir=None):
    """Convert one YCB object into $assets_root/<name>/<name>.xml (+ meshes/, meta json)."""
    ycb_id = spec["ycb_id"]
    cache_dir = os.path.join(assets_root, "_ycb_cache")
    src_obj = _download_extract(ycb_id, cache_dir)
    src_dir = os.path.dirname(src_obj)

    out_dir = os.path.join(assets_root, name)
    mesh_dir = os.path.join(out_dir, "meshes")
    os.makedirs(mesh_dir, exist_ok=True)

    mesh = trimesh.load(src_obj, force="mesh", process=False)
    # Recentre via MuJoCo <mesh refpos> (NOT by re-exporting, which would strip the visual
    # OBJ's texture UVs). refpos is subtracted from every vertex at load, so visual +
    # collision meshes share one recentred frame. Sites/bounds are computed in that frame.
    center = mesh.bounds.mean(axis=0)
    refpos = f"{center[0]:.6f} {center[1]:.6f} {center[2]:.6f}"
    lo, hi = mesh.bounds - center
    ext = hi - lo
    r_xy = float(np.max(np.linalg.norm((mesh.vertices - center)[:, :2], axis=1)))

    # Visual mesh: copy the ORIGINAL textured OBJ verbatim (preserves vt/UVs so MuJoCo maps
    # the YCB texture); strip only mtllib/usemtl so our explicit MJCF material binds instead.
    vis_obj = os.path.join(mesh_dir, "visual.obj")
    with open(src_obj) as fin, open(vis_obj, "w") as fout:
        for line in fin:
            if line.startswith(("mtllib", "usemtl")):
                continue
            fout.write(line)
    tex_name = None
    for cand in ("texture_map.png", "texture_map.jpg", "texture_map.tif"):
        p = os.path.join(src_dir, cand)
        if os.path.exists(p):
            tex_name = "texture.png"
            if cand.endswith(".png"):
                shutil.copy(p, os.path.join(mesh_dir, tex_name))
            else:
                from PIL import Image
                Image.open(p).convert("RGB").save(os.path.join(mesh_dir, tex_name))
            break

    # Collision: coacd convex parts (recentered consistently).
    parts = _coacd_decompose(mesh)
    part_files = []
    for i, part in enumerate(parts):
        pf = os.path.join(mesh_dir, f"collision_{i}.obj")
        part.export(pf)
        part_files.append(os.path.basename(pf))
    print(f"[ycb] {name}: {len(part_files)} collision parts, ext={ext.round(3)}, r_xy={r_xy:.3f}",
          flush=True)

    fr = " ".join(str(x) for x in spec["friction"])
    dens = spec["density"]

    asset_lines = [f'    <mesh file="meshes/visual.obj" name="{name}_vis" refpos="{refpos}"/>']
    for i, pf in enumerate(part_files):
        asset_lines.append(f'    <mesh file="meshes/{pf}" name="{name}_col{i}" refpos="{refpos}"/>')
    mat_line = ""
    if tex_name:
        asset_lines.append(f'    <texture type="2d" file="meshes/{tex_name}" name="{name}_tex"/>')
        asset_lines.append(f'    <material name="{name}_mat" texture="{name}_tex" specular="0.2" shininess="0.2"/>')
        mat_line = f'material="{name}_mat"'

    col_geoms = "\n".join(
        f'        <geom pos="0 0 0" mesh="{name}_col{i}" type="mesh" '
        f'solimp="0.998 0.998 0.001" solref="0.001 1" density="{dens}" '
        f'friction="{fr}" group="0" condim="4"/>'
        for i in range(len(part_files))
    )

    xml = f'''<mujoco model="{name}">
  <asset>
{os.linesep.join(asset_lines)}
  </asset>
  <worldbody>
    <body>
      <body name="object">
        <geom pos="0 0 0" mesh="{name}_vis" type="mesh" {mat_line} contype="0" conaffinity="0" group="1" mass="1e-6"/>
{col_geoms}
      </body>
      <site rgba="0 0 0 0" size="0.005" pos="0 0 {lo[2]:.5f}" name="bottom_site"/>
      <site rgba="0 0 0 0" size="0.005" pos="0 0 {hi[2]:.5f}" name="top_site"/>
      <site rgba="0 0 0 0" size="0.005" pos="{r_xy:.5f} 0 0" name="horizontal_radius_site"/>
    </body>
  </worldbody>
</mujoco>
'''
    xml_path = os.path.join(out_dir, f"{name}.xml")
    with open(xml_path, "w") as f:
        f.write(xml)

    meta = {
        "name": name, "ycb_id": ycb_id,
        "extent": ext.tolist(), "bounds_lo": lo.tolist(), "bounds_hi": hi.tolist(),
        "horizontal_radius": r_xy, "is_container": spec.get("is_container", False),
        # container geometry for the predicate (inner radius ~ 85% outer; rim at top).
        "inner_radius": 0.85 * r_xy, "base_z_offset": float(lo[2]),
        "rim_z_offset": float(hi[2]),
    }
    with open(os.path.join(out_dir, f"{name}_meta.json"), "w") as f:
        json.dump(meta, f, indent=1)
    return xml_path


def ensure_assets(assets_root, names=("banana", "bowl")):
    """Convert all requested YCB objects if not already present; return {name: xml_path}."""
    out = {}
    for name in names:
        xml_path = os.path.join(assets_root, name, f"{name}.xml")
        if not os.path.exists(xml_path):
            xml_path = convert(name, YCB_OBJECTS[name], assets_root)
        out[name] = xml_path
    return out


def load_meta(assets_root, name):
    with open(os.path.join(assets_root, name, f"{name}_meta.json")) as f:
        return json.load(f)


if __name__ == "__main__":
    root = os.environ.get("ROBOLAB_ASSETS_DIR", "/models/robolab_assets")
    print(ensure_assets(root))
