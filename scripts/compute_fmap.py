"""Compute functional map T matrices from cached activations.

Usage:
    python scripts/compute_fmap.py ACTIVATIONS_PATH --source-model ViT-B-16 --target-model ViT-B-16-plus-240 [OPTIONS]

    # All layers:
    python scripts/compute_fmap.py activations.pt --source-model ViT-B-16 --target-model ViT-B-16-plus-240

    # Single layer (for parallel SLURM jobs):
    python scripts/compute_fmap.py activations.pt --source-model ViT-B-16 --target-model ViT-B-16-plus-240 --layer-index 5

    # Specific layer by name:
    python scripts/compute_fmap.py activations.pt --source-model ViT-B-16 --target-model ViT-B-16-plus-240 --layer-name "transformer.resblocks.0.attn.q_proj.in"

    # Custom params:
    python scripts/compute_fmap.py activations.pt --source-model ViT-B-16 --target-model ViT-B-16-plus-240 --num-eigs 80 --k-graph 30

Output structure:
    {output-dir}/{source}_{target}_eigs{E}_k{K}/
        transformer.resblocks.0.attn.q_proj.in.pt
        transformer.resblocks.0.attn.q_proj.out.pt
        ...
"""
import argparse
import os
import re
import sys
import time

import numpy as np
import torch

# Add src to path so we can import fmap_utils
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from merge_and_rebase.rebase.methods.fmap_utils import FM_T

_DEFAULT_FMAP_DIR = "/leonardo_scratch/fast/IscrC_eff-SAM2/HeterogeneousMerging/fmaps"


def _sanitize_layer_name(name: str) -> str:
    """Make a layer name safe for use as a filename (keep dots for readability)."""
    return re.sub(r'[^\w.\-]', '_', name)


def _build_run_dir(base_dir: str, source_model: str, target_model: str,
                   num_eigs: int, k_graph) -> str:
    src = re.sub(r'[^\w\-]', '', source_model.lower())
    tgt = re.sub(r'[^\w\-]', '', target_model.lower())
    k_str = str(k_graph) if k_graph is not None else "auto"
    folder = f"{src}_{tgt}_eigs{num_eigs}_k{k_str}"
    return os.path.join(base_dir, folder)


def main():
    p = argparse.ArgumentParser(description="Compute fmap T matrices from cached activations")
    p.add_argument("activations_path", type=str, help="Path to cached activations .pt file")
    p.add_argument("--source-model", type=str, required=True, help="Source model name (e.g. ViT-B-16)")
    p.add_argument("--target-model", type=str, required=True, help="Target model name (e.g. ViT-B-16-plus-240)")
    p.add_argument("--output-dir", type=str, default=_DEFAULT_FMAP_DIR,
                   help=f"Base output directory (default: {_DEFAULT_FMAP_DIR})")
    p.add_argument("--layer-index", type=int, default=None, help="Compute only this layer index")
    p.add_argument("--layer-name", type=str, default=None, help="Compute only this layer by name")
    p.add_argument("--num-eigs", type=int, default=50)
    p.add_argument("--k-graph", type=int, default=None, help="KNN neighbors (default: 7%% of samples)")
    p.add_argument("--min-dim", type=int, default=10, help="Skip layers with feature dim < this")
    p.add_argument("--max-samples", type=int, default=100000, help="Skip layers with more samples than this")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    # Build run directory
    run_dir = _build_run_dir(args.output_dir, args.source_model, args.target_model,
                             args.num_eigs, args.k_graph)
    os.makedirs(run_dir, exist_ok=True)
    print(f"Output directory: {run_dir}")

    print(f"Loading activations from {args.activations_path}")
    data = torch.load(args.activations_path, map_location="cpu", weights_only=False)

    # Build list of layers to process
    all_keys = list(data.keys())
    print(f"Found {len(all_keys)} layers")

    if args.layer_name is not None:
        if args.layer_name not in data:
            print(f"Layer '{args.layer_name}' not found. Available: {all_keys}")
            sys.exit(1)
        keys = [args.layer_name]
    elif args.layer_index is not None:
        if args.layer_index >= len(all_keys):
            print(f"Layer index {args.layer_index} out of range (0-{len(all_keys)-1})")
            sys.exit(1)
        keys = [all_keys[args.layer_index]]
    else:
        keys = all_keys

    dev = torch.device(args.device)
    n_saved = 0

    for i, key in enumerate(keys):
        # Skip if already computed
        out_file = os.path.join(run_dir, _sanitize_layer_name(key) + ".pt")
        if os.path.isfile(out_file):
            print(f"[{i+1}/{len(keys)}] {key}: already exists, skipping")
            n_saved += 1
            continue

        entry = data[key]
        src = entry["src"]
        tgt = entry["tgt"]
        n_real = entry.get("n_real_samples", entry["n_samples"])
        n_samples = src.shape[0]
        d_src = src.shape[1]
        d_tgt = tgt.shape[1]

        if n_samples > args.max_samples:
            print(f"[{i+1}/{len(keys)}] {key}: skipped (too many points: {n_samples})")
            continue
        if d_src < args.min_dim or d_tgt < args.min_dim:
            print(f"[{i+1}/{len(keys)}] {key}: skipped (dim too small: {d_src}, {d_tgt})")
            continue

        n_anch = min(n_real, n_samples)
        anchors = torch.stack([torch.arange(n_anch), torch.arange(n_anch)], dim=1)
        n_eig = min(args.num_eigs, n_samples - 1)
        k_eff = args.k_graph if args.k_graph is not None else max(int(n_samples * 0.07), 5)

        print(f"[{i+1}/{len(keys)}] {key}: src={tuple(src.shape)} tgt={tuple(tgt.shape)} "
              f"anchors={n_anch} eigs={n_eig} k={k_eff}")

        t0 = time.time()
        try:
            fmap = FM_T(
                src.double(),
                tgt.double(),
                anchors,
                transformation="orthogonal",
                num_eigs=n_eig,
                graph_algo="knn",
                graph_similarity="angular",
                graph_kernel="distance",
                descriptors=("dist_geod",),
                k=k_eff,
                n_descr=1,
                compute_gt_map=False,
                refine=True,
                device=dev,
            )
            T = fmap.T
            if isinstance(T, torch.Tensor):
                T_save = T.float().cpu()
            else:
                T_save = torch.tensor(np.array(T), dtype=torch.float32)

            torch.save(T_save, out_file)
            n_saved += 1

            sim = fmap.get_similarity()
            dt = time.time() - t0
            print(f"  -> T={tuple(T_save.shape)} similarity={sim:.4f} ({dt:.1f}s) saved to {out_file}")
        except Exception as exc:
            print(f"  -> FAILED: {exc}")

    print(f"\nDone. {n_saved} layer transforms in {run_dir}")


if __name__ == "__main__":
    main()
