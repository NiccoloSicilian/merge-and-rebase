"""Spectral functional-map transport.

Computes alignment maps T_in / T_out via the spectral formula:

    T = pinv(X_B) @ Phi_B @ C @ Phi_A^T @ X_A

where C is the functional map between source and target activation manifolds,
Phi_A / Phi_B are their Laplacian eigenbases, and X_A / X_B are the raw
activation matrices.  This avoids the lossy FM-to-p2p nearest-neighbor step
and the extra Procrustes fitting used in standard fmap transport.

Transport formula remains:  t_out^T @ delta_W @ t_in  (weights)
                            delta_b @ t_out           (biases)
"""
from __future__ import annotations

import logging
import os
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from ..base import TensorDict
from ..registry import register
from .theseus import (
    ActivationStore,
    _ZERO_KEYS,
    _apply_transforms_to_visual_delta,
    _iter_with_progress,
    _LayerTransform,
    _merge_split_qkv_state,
    _param_to_module,
    _resolve_device,
    _split_fused_qkv_state,
    _visual_delta_keys,
    _visual_module,
    _visual_state_dict,
    collect_activations,
    _load_activation_registry,
    _save_activation_registry,
)
from .theseus import split_openclip_vit_attn

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Spectral T computation
# ---------------------------------------------------------------------------

def _compute_spectral_T(
    X_A: torch.Tensor,
    X_B: torch.Tensor,
    Phi_A: np.ndarray,
    Phi_B: np.ndarray,
    C: np.ndarray,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Compute spectral transport matrix.

    For each task vector row w_A (a linear functional on the source manifold):
      1. Evaluate on manifold:       f   = X_A @ w_A          (R^n)
      2. Project to spectral basis:  α   = Phi_A^T @ f        (R^k)
      3. Map via functional map:     β   = C @ α              (R^k)  — magnitudes preserved
      4. Reconstruct on target:      g   = Phi_B @ β          (R^n)
      5. Lift to weight space:       w_B = pinv(X_B) @ g      (R^{d_B})

    Composed: T_raw = pinv(X_B) @ Phi_B @ C @ Phi_A^T @ X_A   (d_B, d_A)

    Returns T with shape (d_A, d_B) — Theseus convention for t_in / t_out.
    """
    dev = torch.device(device)

    X_A_d = X_A.double().to(dev)
    X_B_d = X_B.double().to(dev)
    Phi_A_t = torch.tensor(np.asarray(Phi_A), dtype=torch.float64, device=dev)
    Phi_B_t = torch.tensor(np.asarray(Phi_B), dtype=torch.float64, device=dev)
    C_t = torch.tensor(np.asarray(C), dtype=torch.float64, device=dev)

    # Step 1+2: project from weight space to spectral coefficients
    #   project = Phi_A^T @ X_A   (k, d_A)
    project = Phi_A_t.T @ X_A_d

    # Step 3: apply functional map (preserving magnitudes as-is)
    #   mapped = C @ project   (k, d_A)
    mapped = C_t @ project

    # Step 4+5: reconstruct in target weight space
    #   reconstruct = pinv(X_B) @ Phi_B   (d_B, k)
    reconstruct = torch.linalg.pinv(X_B_d) @ Phi_B_t

    # Full transform: T_raw = reconstruct @ mapped   (d_B, d_A)
    T_raw = reconstruct @ mapped

    # Theseus convention: t_in/t_out have shape (d_src, d_tgt)
    # Transport: t_out.T @ delta @ t_in
    return T_raw.T.float().cpu()


def _compute_spectral_transforms_from_activations(
    activation_registry: dict[str, ActivationStore],
    *,
    n_anchors_per_layer: dict[str, int] | None = None,
    n_anchors: int | None = None,
    n_spectral_samples: int | None = None,
    num_eigs: int = 50,
    k_graph: int | None = None,
    device: str | torch.device = "cpu",
    verbose: bool = True,
) -> dict[str, torch.Tensor]:
    """Compute spectral T matrices per registry key.

    Instead of FM_T (which does fmap -> p2p -> Procrustes), we:
    1. Build graphs + eigenbases using the FM base class machinery
    2. Compute functional map C
    3. Derive T = pinv(X_B) @ Phi_B @ C @ Phi_A^T @ X_A
    """
    dev = torch.device(device) if isinstance(device, str) else device
    log_prefix = "[spectral]"
    transforms: dict[str, torch.Tensor] = {}

    for key, store in activation_registry.items():
        src_rows, tgt_rows = store.rows(center=False)
        if src_rows is None or tgt_rows is None:
            if verbose:
                print(f"{log_prefix} {key}: skipped (no raw activations)")
            continue

        n_samples = src_rows.shape[0]
        d_src = src_rows.shape[1]
        d_tgt = tgt_rows.shape[1]

        if n_samples < 3:
            if verbose:
                print(f"{log_prefix} {key}: skipped (only {n_samples} samples)")
            continue
        if n_samples > 100_000:
            if verbose:
                print(f"{log_prefix} {key}: skipped (too many points: {n_samples})")
            continue
        if d_src < 10 or d_tgt < 10:
            if verbose:
                print(f"{log_prefix} {key}: skipped (dim too small: {d_src}, {d_tgt})")
            continue

        # Anchors
        n_real = n_samples
        if n_anchors_per_layer is not None and key in n_anchors_per_layer:
            n_real = n_anchors_per_layer[key]
        n_anch = min(n_real, n_samples)
        if n_anchors is not None and n_anchors < n_anch:
            perm = torch.randperm(n_anch)[:n_anchors]
            perm, _ = perm.sort()
            anchors = torch.stack([perm, perm], dim=1)
            n_anch = n_anchors
        else:
            anchors = torch.stack([torch.arange(n_anch), torch.arange(n_anch)], dim=1)

        n_eig = min(num_eigs, n_samples - 1)
        k_eff = k_graph if k_graph is not None else max(int(n_samples * 0.07), 5)

        if verbose:
            print(
                f"{log_prefix} {key}: src={tuple(src_rows.shape)} tgt={tuple(tgt_rows.shape)} "
                f"anchors={n_anch} eigs={n_eig} k={k_eff}"
            )

        try:
            # Use FM (base class) to get C and eigenvectors without Procrustes
            from .fmap_utils.fm_estimator import FM

            fmap = FM(
                src_rows.double(),
                tgt_rows.double(),
                anchors,
                graph_algo="knn",
                graph_similarity="angular",
                graph_kernel="distance",
                num_eigs=n_eig,
                descriptors=("dist_geod",),
                k=k_eff,
                n_descr=1,
                compute_gt_map=False,
                refine=True,
                device=dev,
            )

            sim = fmap.get_similarity()
            if verbose:
                c_shape = np.array(fmap.C).shape
                print(f"{log_prefix} {key}: fmap C={c_shape} similarity={sim:.4f}")

            # Compute spectral T — use only real samples (exclude interpolated)
            n_use = n_spectral_samples
            if n_use is None and n_anchors_per_layer is not None and key in n_anchors_per_layer:
                n_use = n_anchors_per_layer[key]
            if n_use is not None and n_use < n_samples:
                src_for_T = src_rows[:n_use]
                tgt_for_T = tgt_rows[:n_use]
                eigvecs1_for_T = fmap.eigvecs1[:n_use]
                eigvecs2_for_T = fmap.eigvecs2[:n_use]
            else:
                src_for_T = src_rows
                tgt_for_T = tgt_rows
                eigvecs1_for_T = fmap.eigvecs1
                eigvecs2_for_T = fmap.eigvecs2
            T = _compute_spectral_T(
                src_for_T, tgt_for_T,
                eigvecs1_for_T, eigvecs2_for_T,
                fmap.C, device=dev,
            )
            transforms[key] = T

            if verbose:
                print(f"{log_prefix} {key}: T={tuple(T.shape)}")

        except Exception as exc:
            print(f"{log_prefix} {key}: FAILED — {exc}")

        # Free raw activations
        store.h_a_list.clear()
        store.h_b_list.clear()

    return transforms


# ---------------------------------------------------------------------------
# Precompute per-layer transforms
# ---------------------------------------------------------------------------

def _precompute_spectral_transforms(
    *,
    target_model: torch.nn.Module,
    target_visual_base: Mapping[str, torch.Tensor],
    visual_delta: Mapping[str, torch.Tensor],
    spectral_transforms: Mapping[str, torch.Tensor],
    show_progress: bool,
    method_name: str,
) -> dict[str, _LayerTransform]:
    """Map spectral T matrices to per-parameter _LayerTransform entries."""
    transforms_by_key: dict[str, _LayerTransform] = {}
    t_out_cache: dict[str, torch.Tensor] = {}
    visual_model = _visual_module(target_model)
    param_to_mod = _param_to_module(visual_model)

    items = _iter_with_progress(
        visual_delta.items(),
        total=len(visual_delta),
        desc=f"{method_name}.prepare: compute transforms",
        enabled=show_progress,
    )
    for key, delta_source in items:
        if key not in target_visual_base:
            continue

        if key in _ZERO_KEYS:
            transforms_by_key[key] = _LayerTransform(kind="zero")
            continue

        module_name = param_to_mod.get(key, key.rsplit(".", 1)[0] if "." in key else "")
        if key == "proj":
            in_key = "ln_post.out"
            out_key = ".out"
        else:
            in_key = f"{module_name}.in"
            out_key = f"{module_name}.out"

        if delta_source.ndim == 2:
            t_in = spectral_transforms.get(in_key)
            t_out = spectral_transforms.get(out_key)
            if t_in is not None and t_out is not None:
                transforms_by_key[key] = _LayerTransform(kind="weight", t_in=t_in, t_out=t_out)
            else:
                transforms_by_key[key] = _LayerTransform(kind="weight")
            continue

        if delta_source.ndim == 1:
            if key.endswith(".bias"):
                weight_key = f"{key[: -len('.bias')]}.weight"
                weight_transform = transforms_by_key.get(weight_key)
                if weight_transform is not None and weight_transform.t_out is not None:
                    transforms_by_key[key] = _LayerTransform(kind="bias", t_out=weight_transform.t_out)
                    continue

            t_out = t_out_cache.get(out_key) or spectral_transforms.get(out_key)
            if t_out is not None:
                t_out_cache[out_key] = t_out
                transforms_by_key[key] = _LayerTransform(kind="bias", t_out=t_out)
            else:
                transforms_by_key[key] = _LayerTransform(kind="bias")
            continue

        transforms_by_key[key] = _LayerTransform(kind="unsupported")

    return transforms_by_key


# ---------------------------------------------------------------------------
# Method class
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SpectralTransport:
    """Transport task vectors via spectral functional map alignment.

    Uses T = pinv(X_B) @ Phi_B @ C @ Phi_A^T @ X_A instead of
    the standard fmap -> p2p -> Procrustes pipeline.
    """

    name: str = "spectral"

    def prepare(
        self,
        *,
        source_model: torch.nn.Module,
        target_model: torch.nn.Module,
        source_dataloader: Iterable[Any] | None = None,
        target_dataloader: Iterable[Any] | None = None,
        activation_source_model: torch.nn.Module | None = None,
        activation_target_model: torch.nn.Module | None = None,
        target_base: Mapping[str, torch.Tensor] | None = None,
        delta: Mapping[str, torch.Tensor] | None = None,
        device: str = "cuda",
        seq_align: str = "interpolate2d",
        center_acts: bool = False,
        n_batches: int | None = None,
        num_batches: int | None = None,
        seed: int = 0,
        batch_size: int | None = None,
        patch_qkv: bool = True,
        n_interpolations: int = 0,
        num_eigs: int = 50,
        k_graph: int | None = None,
        n_anchors: int | None = None,
        n_spectral_samples: int | None = None,
        activations_path: str | None = None,
        fmap_transforms_path: str | None = None,
        verbose: bool = True,
        show_progress: bool = True,
        **kwargs,
    ) -> dict[str, Any]:
        split_qkv = kwargs.pop("split_qkv", None)
        if split_qkv is not None:
            patch_qkv = bool(split_qkv)
        del kwargs
        n_interpolations = int(n_interpolations)

        if n_batches is None:
            n_batches = num_batches

        log_prefix = f"[{self.name}]"
        dev = _resolve_device(device)

        # Patch fused QKV
        split_fused_qkv = False
        if patch_qkv:
            s_splits = split_openclip_vit_attn(source_model)
            t_splits = split_openclip_vit_attn(target_model)
            split_fused_qkv = bool(s_splits or t_splits)
            if split_fused_qkv and verbose:
                print(f"{log_prefix} prepare: split fused QKV in {s_splits + t_splits} modules")

        # Use activation models if provided
        act_src = activation_source_model if activation_source_model is not None else source_model
        act_tgt = activation_target_model if activation_target_model is not None else target_model

        # Collect or load activations
        activation_registry: dict[str, ActivationStore] = {}
        n_real_samples_per_layer: dict[str, int] = {}

        _act_path = activations_path if activations_path else None

        if _act_path and os.path.isfile(_act_path):
            activation_registry, n_real_samples_per_layer = _load_activation_registry(_act_path)
            if verbose:
                print(f"{log_prefix} prepare: loaded cached activations ({len(activation_registry)} layers) from {_act_path}")
        else:
            if source_dataloader is None or target_dataloader is None:
                raise ValueError("spectral transport requires source_dataloader and target_dataloader")
            if verbose:
                print(f"{log_prefix} prepare: collecting activations")
            activation_registry = collect_activations(
                source_model=act_src,
                target_model=act_tgt,
                source_dataloader=source_dataloader,
                target_dataloader=target_dataloader,
                device=dev,
                n_batches=n_batches,
                seed=int(seed),
                batch_size=batch_size,
                seq_align=seq_align,
                n_interpolations=n_interpolations,
                store_raw=True,
                show_progress=bool(show_progress),
            )
            for key, store in activation_registry.items():
                if store.h_a_list:
                    total = sum(t.shape[0] for t in store.h_a_list)
                    interp_total = n_interpolations
                    n_real_samples_per_layer[key] = max(total - interp_total, 0)
                else:
                    n_real_samples_per_layer[key] = store.n_samples

            if _act_path:
                _save_activation_registry(activation_registry, _act_path, n_real_samples_per_layer)

        # Compute or load spectral transforms
        _fmap_path = fmap_transforms_path if fmap_transforms_path else None
        spectral_transforms: dict[str, torch.Tensor] = {}

        if _fmap_path and os.path.isdir(_fmap_path):
            # Load precomputed fmap components and derive spectral T
            n_loaded = 0
            for fname in os.listdir(_fmap_path):
                if not fname.endswith(".pt"):
                    continue
                layer_key = fname[:-3]
                fmap_data = torch.load(
                    os.path.join(_fmap_path, fname),
                    map_location="cpu", weights_only=False,
                )
                # New format: dict with C, eigvecs, etc.
                if isinstance(fmap_data, dict) and "C" in fmap_data:
                    # Need raw activations to compute spectral T
                    store = activation_registry.get(layer_key)
                    if store is not None:
                        src_rows, tgt_rows = store.rows(center=False)
                        if src_rows is not None and tgt_rows is not None:
                            # Use only real samples (exclude interpolated)
                            n_use = n_spectral_samples
                            if n_use is None:
                                n_real = n_real_samples_per_layer.get(layer_key)
                                if n_real is not None:
                                    n_use = n_real
                            if n_use is not None and n_use < src_rows.shape[0]:
                                src_rows = src_rows[:n_use]
                                tgt_rows = tgt_rows[:n_use]
                            T = _compute_spectral_T(
                                src_rows, tgt_rows,
                                fmap_data["eigvecs1"].numpy()[:n_use] if n_use is not None else fmap_data["eigvecs1"].numpy(),
                                fmap_data["eigvecs2"].numpy()[:n_use] if n_use is not None else fmap_data["eigvecs2"].numpy(),
                                fmap_data["C"].numpy(),
                                device=dev,
                            )
                            spectral_transforms[layer_key] = T
                            n_loaded += 1
                            if verbose:
                                sim = fmap_data.get("similarity", 0.0)
                                n_pts = src_rows.shape[0]
                                print(f"{log_prefix} {layer_key}: loaded fmap (sim={sim:.4f}) samples={n_pts} -> T={tuple(T.shape)}")
                    else:
                        if verbose:
                            print(f"{log_prefix} {layer_key}: fmap loaded but no activations for spectral T")
                else:
                    # Legacy format: raw T tensor (from theseus fmap)
                    spectral_transforms[layer_key] = fmap_data
                    n_loaded += 1
            if verbose:
                print(f"{log_prefix} prepare: built spectral transforms for {n_loaded} layers from {_fmap_path}")
        else:
            if verbose:
                print(f"{log_prefix} prepare: computing spectral transforms from activations")
            spectral_transforms = _compute_spectral_transforms_from_activations(
                activation_registry,
                n_anchors_per_layer=n_real_samples_per_layer,
                n_anchors=n_anchors,
                n_spectral_samples=n_spectral_samples,
                num_eigs=int(num_eigs),
                k_graph=k_graph,
                device=device,
                verbose=bool(verbose),
            )

        if verbose:
            print(f"{log_prefix} prepare: spectral transforms for {len(spectral_transforms)} layers")

        # Precompute per-parameter transforms
        transforms_by_key: dict[str, _LayerTransform] = {}
        if target_base is not None and delta is not None:
            visual_key_map = _visual_delta_keys(delta)
            source_visual_base = _visual_state_dict(source_model.state_dict())
            target_visual_base = _visual_state_dict(target_base)
            visual_delta = {
                stripped_key: delta[original_key]
                for stripped_key, original_key in visual_key_map.items()
                if stripped_key in target_visual_base
            }

            if split_fused_qkv:
                target_visual_base = _split_fused_qkv_state(target_visual_base)
                visual_delta = _split_fused_qkv_state(visual_delta)

            transforms_by_key = _precompute_spectral_transforms(
                target_model=target_model,
                target_visual_base=target_visual_base,
                visual_delta=visual_delta,
                spectral_transforms=spectral_transforms,
                show_progress=bool(show_progress),
                method_name=self.name,
            )

        return {
            "transforms_by_key": transforms_by_key,
            "split_fused_qkv": split_fused_qkv,
        }

    def apply(
        self,
        prepared: Mapping[str, Any],
        *,
        target_base: Mapping[str, torch.Tensor],
        delta: Mapping[str, torch.Tensor],
        strict: bool = False,
        verbose: bool = True,
        show_progress: bool = True,
        **kwargs,
    ) -> TensorDict:
        del kwargs
        log_prefix = f"[{self.name}]"

        transforms_by_key = prepared.get("transforms_by_key")
        if transforms_by_key is None:
            raise ValueError("Spectral prepared payload is missing 'transforms_by_key'.")

        visual_key_map = _visual_delta_keys(delta)
        target_visual_base = _visual_state_dict(target_base)

        visual_delta = {
            stripped_key: delta[original_key]
            for stripped_key, original_key in visual_key_map.items()
            if stripped_key in target_visual_base
        }

        split_fused_qkv = bool(prepared.get("split_fused_qkv", False))
        if split_fused_qkv:
            target_visual_base_work = _split_fused_qkv_state(target_visual_base)
            visual_delta_work = _split_fused_qkv_state(visual_delta)
        else:
            target_visual_base_work = target_visual_base
            visual_delta_work = visual_delta

        aligned_visual = _apply_transforms_to_visual_delta(
            target_visual_base=target_visual_base_work,
            visual_delta=visual_delta_work,
            transforms_by_key=transforms_by_key,
            show_progress=bool(show_progress),
            method_name=self.name,
        )

        if split_fused_qkv:
            aligned_visual = _merge_split_qkv_state(aligned_visual, reference=target_visual_base)

        out: TensorDict = {}
        processed: set[str] = set()

        for stripped_key, original_key in visual_key_map.items():
            if original_key not in target_base:
                continue
            if stripped_key in aligned_visual:
                out[original_key] = aligned_visual[stripped_key].to(
                    dtype=target_base[original_key].dtype,
                    device=target_base[original_key].device,
                )
            else:
                out[original_key] = torch.zeros_like(target_base[original_key])
            processed.add(original_key)

        for key in delta:
            if key in processed or key not in target_base:
                continue
            out[key] = torch.zeros_like(target_base[key])

        if verbose:
            print(f"{log_prefix} apply: done (transported_keys={len(out)})")

        return out

    def transport(
        self,
        *,
        source_base: Mapping[str, torch.Tensor],
        target_base: Mapping[str, torch.Tensor],
        delta: Mapping[str, torch.Tensor],
        strict: bool = False,
        source_model: torch.nn.Module | None = None,
        target_model: torch.nn.Module | None = None,
        source_dataloader: Iterable[Any] | None = None,
        target_dataloader: Iterable[Any] | None = None,
        activation_source_model: torch.nn.Module | None = None,
        activation_target_model: torch.nn.Module | None = None,
        device: str = "cuda",
        seq_align: str = "interpolate2d",
        center_acts: bool = False,
        n_batches: int | None = None,
        num_batches: int | None = None,
        seed: int = 0,
        batch_size: int | None = None,
        patch_qkv: bool = True,
        n_interpolations: int = 0,
        num_eigs: int = 50,
        k_graph: int | None = None,
        n_anchors: int | None = None,
        n_spectral_samples: int | None = None,
        activations_path: str | None = None,
        fmap_transforms_path: str | None = None,
        prepared: Mapping[str, Any] | None = None,
        verbose: bool = True,
        show_progress: bool = True,
        **kwargs,
    ) -> TensorDict:
        del source_base

        if n_batches is None:
            n_batches = num_batches

        if prepared is None:
            if source_model is None or target_model is None:
                raise ValueError("spectral transport requires both source_model and target_model.")
            if source_dataloader is None or target_dataloader is None:
                raise ValueError("spectral transport requires both source_dataloader and target_dataloader.")

            prepared = self.prepare(
                source_model=source_model,
                target_model=target_model,
                source_dataloader=source_dataloader,
                target_dataloader=target_dataloader,
                activation_source_model=activation_source_model,
                activation_target_model=activation_target_model,
                target_base=target_base,
                delta=delta,
                device=device,
                seq_align=seq_align,
                center_acts=bool(center_acts),
                n_batches=n_batches,
                seed=int(seed),
                batch_size=batch_size,
                patch_qkv=patch_qkv,
                n_interpolations=int(n_interpolations),
                num_eigs=int(num_eigs),
                k_graph=k_graph,
                n_anchors=n_anchors,
                n_spectral_samples=n_spectral_samples,
                activations_path=activations_path,
                fmap_transforms_path=fmap_transforms_path,
                verbose=bool(verbose),
                show_progress=bool(show_progress),
                **kwargs,
            )

        return self.apply(
            prepared,
            target_base=target_base,
            delta=delta,
            strict=strict,
            verbose=bool(verbose),
            show_progress=bool(show_progress),
        )


register(SpectralTransport())
