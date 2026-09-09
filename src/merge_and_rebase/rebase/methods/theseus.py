from __future__ import annotations

import logging
import os
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from ...models.patch_openclip_attention import merge_openclip_vit_attn, split_openclip_vit_attn
from ..base import TensorDict
from ..registry import register

logger = logging.getLogger(__name__)

_VISUAL_PREFIX = "visual."
_ZERO_KEYS = {"class_embedding", "positional_embedding", "conv1.weight"}
_FUSED_IN_PROJ_WEIGHT = ".attn.in_proj_weight"
_FUSED_IN_PROJ_BIAS = ".attn.in_proj_bias"
_Q_PROJ_WEIGHT = ".attn.q_proj.weight"
_K_PROJ_WEIGHT = ".attn.k_proj.weight"
_V_PROJ_WEIGHT = ".attn.v_proj.weight"
_Q_PROJ_BIAS = ".attn.q_proj.bias"
_K_PROJ_BIAS = ".attn.k_proj.bias"
_V_PROJ_BIAS = ".attn.v_proj.bias"

_ACTIVATION_COVARIANCE_MODES = {"activation", "activations"}
_DATA_FREE_COVARIANCE_MODES = {"data_free", "data-free", "weight", "weights", "weight_space", "weight-space"}


def _resolve_device(device: str | torch.device) -> torch.device:
    dev = torch.device(device)
    if dev.type == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return dev


def _extract_model_inputs(batch: Any) -> torch.Tensor:
    if torch.is_tensor(batch):
        return batch
    if isinstance(batch, Mapping):
        for key in ("pixel_values", "images", "image", "inputs", "x"):
            value = batch.get(key, None)
            if torch.is_tensor(value):
                return value
    if isinstance(batch, (tuple, list)) and batch:
        first = batch[0]
        if torch.is_tensor(first):
            return first
    raise TypeError("Unsupported batch format for Theseus calibration.")


def _extract_output_tensor(output: Any) -> torch.Tensor:
    if torch.is_tensor(output):
        return output
    if isinstance(output, (tuple, list)) and output:
        first = output[0]
        if torch.is_tensor(first):
            return first
    raise TypeError("Unsupported module output while collecting Theseus activations.")


def _encode_image(model: torch.nn.Module, images: torch.Tensor) -> torch.Tensor:
    if hasattr(model, "encode_image") and callable(model.encode_image):
        return model.encode_image(images)
    if hasattr(model, "visual") and callable(model.visual):
        return model.visual(images)
    return model(images)


def _visual_module(model: torch.nn.Module) -> torch.nn.Module:
    return model.visual if hasattr(model, "visual") else model


def _has_fused_mha(visual: torch.nn.Module) -> bool:
    transformer = getattr(visual, "transformer", None)
    resblocks = getattr(transformer, "resblocks", None)
    if resblocks is None:
        return False
    for block in resblocks:
        if isinstance(getattr(block, "attn", None), nn.MultiheadAttention):
            return True
    return False


def _split_fused_qkv_if_needed(model: torch.nn.Module) -> int:
    visual = _visual_module(model)
    if not _has_fused_mha(visual):
        return 0

    ref_param = next(visual.parameters(), None)
    ref_device = ref_param.device if ref_param is not None else torch.device("cpu")
    ref_dtype = ref_param.dtype if ref_param is not None else None

    n_patched = int(
        split_openclip_vit_attn(
            visual,
            proj_dropout=0.0,
            attn_impl="softmax",
        )
    )

    if n_patched > 0:
        if ref_dtype is None:
            visual.to(device=ref_device)
        else:
            visual.to(device=ref_device, dtype=ref_dtype)

    return n_patched


def _visual_state_dict(sd: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    visual = {key[len(_VISUAL_PREFIX) :]: value for key, value in sd.items() if key.startswith(_VISUAL_PREFIX)}
    return visual if visual else dict(sd)


def _visual_delta_keys(delta: Mapping[str, torch.Tensor]) -> dict[str, str]:
    visual = {key[len(_VISUAL_PREFIX) :]: key for key in delta if key.startswith(_VISUAL_PREFIX)}
    if visual:
        return visual
    return {key: key for key in delta}


def _split_fused_qkv_state(state: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {}
    for key, value in state.items():
        if key.endswith(_FUSED_IN_PROJ_WEIGHT) and value.ndim == 2 and value.shape[0] % 3 == 0:
            base = key[: -len(_FUSED_IN_PROJ_WEIGHT)]
            c = value.shape[0] // 3
            out[f"{base}{_Q_PROJ_WEIGHT}"] = value[:c, :]
            out[f"{base}{_K_PROJ_WEIGHT}"] = value[c : 2 * c, :]
            out[f"{base}{_V_PROJ_WEIGHT}"] = value[2 * c :, :]
            continue

        if key.endswith(_FUSED_IN_PROJ_BIAS) and value.ndim == 1 and value.shape[0] % 3 == 0:
            base = key[: -len(_FUSED_IN_PROJ_BIAS)]
            c = value.shape[0] // 3
            out[f"{base}{_Q_PROJ_BIAS}"] = value[:c]
            out[f"{base}{_K_PROJ_BIAS}"] = value[c : 2 * c]
            out[f"{base}{_V_PROJ_BIAS}"] = value[2 * c :]
            continue

        out[key] = value
    return out


def _merge_split_qkv_state(
    state: Mapping[str, torch.Tensor],
    *,
    reference: Mapping[str, torch.Tensor] | None = None,
) -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = dict(state)

    def _merge_triplet(q_suffix: str, k_suffix: str, v_suffix: str, fused_suffix: str) -> None:
        prefixes: set[str] = set()
        for k in tuple(out.keys()):
            if k.endswith(q_suffix):
                prefixes.add(k[: -len(q_suffix)])
            elif k.endswith(k_suffix):
                prefixes.add(k[: -len(k_suffix)])
            elif k.endswith(v_suffix):
                prefixes.add(k[: -len(v_suffix)])

        for p in prefixes:
            qk = f"{p}{q_suffix}"
            kk = f"{p}{k_suffix}"
            vk = f"{p}{v_suffix}"
            fused = f"{p}{fused_suffix}"
            if qk not in out or kk not in out or vk not in out:
                continue
            if reference is not None and fused not in reference:
                continue

            merged = torch.cat([out[qk], out[kk], out[vk]], dim=0)
            out[fused] = merged
            del out[qk]
            del out[kk]
            del out[vk]

    _merge_triplet(_Q_PROJ_WEIGHT, _K_PROJ_WEIGHT, _V_PROJ_WEIGHT, _FUSED_IN_PROJ_WEIGHT)
    _merge_triplet(_Q_PROJ_BIAS, _K_PROJ_BIAS, _V_PROJ_BIAS, _FUSED_IN_PROJ_BIAS)
    return out


def _is_square(n: int) -> bool:
    if n <= 0:
        return False
    r = int(n**0.5)
    return r * r == n


def _standardize_tokens(x: torch.Tensor, *, batch_size: int) -> torch.Tensor:
    if x.ndim == 1:
        return x.view(1, 1, -1)
    if x.ndim == 2:
        return x.unsqueeze(1)
    if x.ndim == 3:
        if x.shape[0] == batch_size:
            return x
        if x.shape[1] == batch_size:
            return x.transpose(0, 1)
        return x
    if x.ndim == 4:
        return x
    return x.reshape(batch_size, -1, x.shape[-1])


def _to_tokens(x: torch.Tensor, *, batch_size: int) -> torch.Tensor:
    x = _standardize_tokens(x, batch_size=batch_size)
    if x.ndim == 4:
        return x.permute(0, 2, 3, 1).reshape(x.shape[0], -1, x.shape[1])
    if x.ndim == 3:
        return x
    return x.reshape(batch_size, -1, x.shape[-1])


def _interp_linear_tokens(tokens: torch.Tensor, target_tokens: int) -> torch.Tensor:
    if tokens.shape[1] == target_tokens:
        return tokens
    tokens_t = tokens.transpose(1, 2)
    tokens_t = F.interpolate(tokens_t, size=target_tokens, mode="linear", align_corners=False)
    return tokens_t.transpose(1, 2)


def _interp_2d_tokens(tokens: torch.Tensor, target_tokens: int) -> torch.Tensor:
    if tokens.shape[1] == target_tokens:
        return tokens

    has_cls = _is_square(tokens.shape[1] - 1) and _is_square(target_tokens - 1)
    cls_token: torch.Tensor | None = None
    patch_tokens = tokens
    target_patch_tokens = target_tokens

    if has_cls:
        cls_token = tokens[:, :1, :]
        patch_tokens = tokens[:, 1:, :]
        target_patch_tokens = target_tokens - 1

    if not _is_square(patch_tokens.shape[1]) or not _is_square(target_patch_tokens):
        resized = _interp_linear_tokens(patch_tokens, target_patch_tokens)
        return torch.cat([cls_token, resized], dim=1) if cls_token is not None else resized

    src_side = int(patch_tokens.shape[1] ** 0.5)
    tgt_side = int(target_patch_tokens**0.5)
    x = patch_tokens.reshape(tokens.shape[0], src_side, src_side, patch_tokens.shape[-1]).permute(0, 3, 1, 2)
    x = F.interpolate(x, size=(tgt_side, tgt_side), mode="bilinear", align_corners=False)
    resized = x.permute(0, 2, 3, 1).reshape(tokens.shape[0], tgt_side * tgt_side, patch_tokens.shape[-1])
    return torch.cat([cls_token, resized], dim=1) if cls_token is not None else resized


def _align_features(
    source_feat: torch.Tensor,
    target_feat: torch.Tensor,
    *,
    mode: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    if source_feat.shape[0] != target_feat.shape[0]:
        raise ValueError(
            f"Theseus calibration expects aligned batch sizes. Got {source_feat.shape[0]} and {target_feat.shape[0]}."
        )

    source_tokens = _to_tokens(source_feat, batch_size=int(source_feat.shape[0]))
    target_tokens = _to_tokens(target_feat, batch_size=int(target_feat.shape[0]))

    if mode == "cls":
        source_tokens = source_tokens[:, :1, :]
        target_tokens = target_tokens[:, :1, :]
    elif mode == "mean":
        source_tokens = source_tokens.mean(dim=1, keepdim=True)
        target_tokens = target_tokens.mean(dim=1, keepdim=True)
    elif mode in {"interpolate2d", "interpolate_2d"}:
        source_tokens = _interp_2d_tokens(source_tokens, int(target_tokens.shape[1]))
    elif mode == "interpolate":
        source_tokens = _interp_linear_tokens(source_tokens, int(target_tokens.shape[1]))

    return source_tokens.reshape(-1, source_tokens.shape[-1]), target_tokens.reshape(-1, target_tokens.shape[-1])


class ActivationStore:
    """Streaming activation statistics with optional Gram and raw storage."""

    def __init__(self, *, store_raw: bool = False, store_a_gram: bool = False, store_b_gram: bool = False) -> None:
        self.store_raw = bool(store_raw)
        self.store_a_gram = bool(store_a_gram)
        self.store_b_gram = bool(store_b_gram)

        self.at_b: torch.Tensor | None = None
        self.at_a: torch.Tensor | None = None
        self.bt_b: torch.Tensor | None = None
        self.sum_a: torch.Tensor | None = None
        self.sum_b: torch.Tensor | None = None
        self.n_samples = 0

        self.h_a_list: list[torch.Tensor] = []
        self.h_b_list: list[torch.Tensor] = []

    def update(self, batch_a: torch.Tensor, batch_b: torch.Tensor) -> None:
        a = batch_a.detach().cpu().to(torch.float64).clone()
        b = batch_b.detach().cpu().to(torch.float64).clone()

        if self.store_raw:
            self.h_a_list.append(a.float())
            self.h_b_list.append(b.float())

        if self.at_b is None:
            self.at_b = (a.T @ b).clone()
            self.sum_a = a.sum(dim=0).clone()
            self.sum_b = b.sum(dim=0).clone()
            if self.store_a_gram:
                self.at_a = (a.T @ a).clone()
            if self.store_b_gram:
                self.bt_b = (b.T @ b).clone()
        else:
            self.at_b += a.T @ b
            self.sum_a += a.sum(dim=0)
            self.sum_b += b.sum(dim=0)
            if self.store_a_gram and self.at_a is not None:
                self.at_a += a.T @ a
            if self.store_b_gram and self.bt_b is not None:
                self.bt_b += b.T @ b

        self.n_samples += int(a.shape[0])

    def rows(self, *, center: bool = False) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        if not self.store_raw or not self.h_a_list:
            return None, None
        source = torch.cat(self.h_a_list, dim=0)
        target = torch.cat(self.h_b_list, dim=0)
        if center:
            source = source - source.mean(dim=0, keepdim=True)
            target = target - target.mean(dim=0, keepdim=True)
        return source, target

    def get_covariance(self, *, center: bool = False, epsilon: float = 0.0) -> torch.Tensor | None:
        if self.at_b is None:
            return None
        cov = self.at_b.clone()
        if center:
            assert self.sum_a is not None and self.sum_b is not None
            mu_a = self.sum_a / self.n_samples
            mu_b = self.sum_b / self.n_samples
            cov = cov - self.n_samples * torch.outer(mu_a, mu_b)
        if epsilon > 0 and cov.shape[0] == cov.shape[1]:
            cov = cov + epsilon * torch.eye(cov.shape[0], dtype=cov.dtype, device=cov.device)
        return cov

    def get_a_gram(self, *, center: bool = False, epsilon: float = 0.0) -> torch.Tensor | None:
        if self.at_a is None:
            return None
        gram = self.at_a.clone()
        if center:
            assert self.sum_a is not None
            mu_a = self.sum_a / self.n_samples
            gram = gram - self.n_samples * torch.outer(mu_a, mu_a)
        if epsilon > 0 and gram.shape[0] == gram.shape[1]:
            gram = gram + epsilon * torch.eye(gram.shape[0], dtype=gram.dtype, device=gram.device)
        return gram

    def get_b_gram(self, *, center: bool = False, epsilon: float = 0.0) -> torch.Tensor | None:
        if self.bt_b is None:
            return None
        gram = self.bt_b.clone()
        if center:
            assert self.sum_b is not None
            mu_b = self.sum_b / self.n_samples
            gram = gram - self.n_samples * torch.outer(mu_b, mu_b)
        if epsilon > 0 and gram.shape[0] == gram.shape[1]:
            gram = gram + epsilon * torch.eye(gram.shape[0], dtype=gram.dtype, device=gram.device)
        return gram


class _ActivationHook:
    def __init__(self, model: torch.nn.Module):
        self.model = _visual_module(model)
        self.inputs: dict[str, torch.Tensor] = {}
        self.outputs: dict[str, torch.Tensor] = {}
        self.handles: list[Any] = []
        self._register_hooks()

    def _register_hooks(self) -> None:
        self.handles.append(self.model.register_forward_hook(self._make_hook("")))
        for name, module in self.model.named_modules():
            if name == "":
                continue
            if list(module.parameters(recurse=False)):
                self.handles.append(module.register_forward_hook(self._make_hook(name)))

    def _make_hook(self, name: str):
        def hook_fn(_module: torch.nn.Module, inputs: tuple[Any, ...], output: Any) -> None:
            inp = inputs[0] if isinstance(inputs, (tuple, list)) and inputs else inputs
            if torch.is_tensor(inp):
                self.inputs[name] = inp.detach().cpu()
            try:
                out = _extract_output_tensor(output)
            except TypeError:
                out = None
            if out is not None and torch.is_tensor(out):
                self.outputs[name] = out.detach().cpu()

        return hook_fn

    def clear(self) -> None:
        self.inputs.clear()
        self.outputs.clear()

    def remove(self) -> None:
        for handle in self.handles:
            handle.remove()


@torch.no_grad()
def collect_activations(
    source_model: torch.nn.Module,
    target_model: torch.nn.Module,
    source_dataloader: Iterable[Any],
    target_dataloader: Iterable[Any],
    *,
    device: str | torch.device,
    seq_align: str,
    n_batches: int | None,
    seed: int = 0,
    batch_size: int | None = None,
    store_raw: bool = False,
    store_a_gram: bool = False,
    store_b_gram: bool = False,
) -> dict[str, ActivationStore]:
    registry: dict[str, ActivationStore] = {}
    source_hook = _ActivationHook(source_model)
    target_hook = _ActivationHook(target_model)
    dev = _resolve_device(device)

    try:
        iterator = _iter_random_dataset_batches(
            source_dataloader,
            target_dataloader,
            n_batches=n_batches,
            seed=seed,
            batch_size=batch_size,
        )
        if iterator is None:
            iterator = zip(source_dataloader, target_dataloader, strict=True)

        for idx, (source_batch, target_batch) in enumerate(iterator):
            if n_batches is not None and idx >= n_batches:
                break

            source_imgs = _extract_model_inputs(source_batch).to(dev)
            target_imgs = _extract_model_inputs(target_batch).to(dev)
            if source_imgs.shape[0] != target_imgs.shape[0]:
                raise ValueError(
                    "Theseus calibration expects aligned batch sizes. "
                    f"Got {source_imgs.shape[0]} and {target_imgs.shape[0]}."
                )

            _encode_image(source_model, source_imgs)
            _encode_image(target_model, target_imgs)

            common_inputs = set(source_hook.inputs.keys()) & set(target_hook.inputs.keys())
            common_outputs = set(source_hook.outputs.keys()) & set(target_hook.outputs.keys())

            for key in common_inputs:
                src_rows, tgt_rows = _align_features(source_hook.inputs[key], target_hook.inputs[key], mode=seq_align)
                reg_key = f"{key}.in"
                registry.setdefault(
                    reg_key,
                    ActivationStore(
                        store_raw=store_raw,
                        store_a_gram=store_a_gram,
                        store_b_gram=store_b_gram,
                    ),
                ).update(src_rows, tgt_rows)

            for key in common_outputs:
                src_rows, tgt_rows = _align_features(source_hook.outputs[key], target_hook.outputs[key], mode=seq_align)
                reg_key = f"{key}.out"
                registry.setdefault(
                    reg_key,
                    ActivationStore(
                        store_raw=store_raw,
                        store_a_gram=store_a_gram,
                        store_b_gram=store_b_gram,
                    ),
                ).update(src_rows, tgt_rows)

            source_hook.clear()
            target_hook.clear()
    finally:
        source_hook.remove()
        target_hook.remove()

    return registry


def _save_activation_registry(
    registry: dict[str, ActivationStore],
    path: str,
    n_real_samples_per_layer: dict[str, int] | None = None,
) -> None:
    """Save raw activations from the registry to disk."""
    import os
    data: dict[str, dict] = {}
    for key, store in registry.items():
        src, tgt = store.rows(center=False)
        if src is not None and tgt is not None:
            data[key] = {
                "src": src,
                "tgt": tgt,
                "n_samples": store.n_samples,
                "at_b": store.at_b,
                "sum_a": store.sum_a,
                "sum_b": store.sum_b,
                "at_a": store.at_a,
                "bt_b": store.bt_b,
                "n_real_samples": n_real_samples_per_layer.get(key, store.n_samples) if n_real_samples_per_layer else store.n_samples,
            }
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save(data, path)
    print(f"[theseus] Saved activation registry ({len(data)} layers) to {path}")


def _load_activation_registry(path: str) -> tuple[dict[str, ActivationStore], dict[str, int]]:
    """Load raw activations from disk into an activation registry.

    Returns (registry, n_real_samples_per_layer).
    """
    data = torch.load(path, map_location="cpu", weights_only=False)
    registry: dict[str, ActivationStore] = {}
    n_real: dict[str, int] = {}
    for key, entry in data.items():
        store = ActivationStore(store_raw=True)
        store.h_a_list = [entry["src"]]
        store.h_b_list = [entry["tgt"]]
        store.n_samples = entry["n_samples"]
        store.at_b = entry["at_b"]
        store.sum_a = entry["sum_a"]
        store.sum_b = entry["sum_b"]
        store.at_a = entry.get("at_a")
        store.bt_b = entry.get("bt_b")
        n_real[key] = entry.get("n_real_samples", store.n_samples)
        registry[key] = store
    print(f"[theseus] Loaded activation registry ({len(registry)} layers) from {path}")
    return registry, n_real


def _slerp_rows(a: torch.Tensor, b: torch.Tensor, t: torch.Tensor, *, eps: float = 1e-7) -> torch.Tensor:
    """Spherical interpolation between corresponding rows of a and b.

    Activations are not unit norm, so the direction is interpolated on the unit
    sphere while the magnitude is interpolated linearly. Rows whose directions
    are (anti)parallel, or which have a vanishing norm, fall back to the linear
    path, where slerp is undefined.
    """
    a = a.to(torch.float32)
    b = b.to(torch.float32)
    na = a.norm(dim=1, keepdim=True)
    nb = b.norm(dim=1, keepdim=True)

    ua = a / na.clamp_min(eps)
    ub = b / nb.clamp_min(eps)

    cos = (ua * ub).sum(dim=1, keepdim=True).clamp(-1.0 + eps, 1.0 - eps)
    omega = torch.acos(cos)
    sin_omega = torch.sin(omega)

    coef_a = torch.sin((1.0 - t) * omega) / sin_omega
    coef_b = torch.sin(t * omega) / sin_omega
    direction = coef_a * ua + coef_b * ub

    # magnitude moves linearly; direction moves along the great circle
    out = direction * ((1.0 - t) * na + t * nb)

    degenerate = (sin_omega.abs() < eps) | (na < eps) | (nb < eps)
    if degenerate.any():
        linear = (1.0 - t) * a + t * b
        out = torch.where(degenerate, linear, out)
    return out


def _augment_registry_with_interpolations(
    registry: dict[str, ActivationStore],
    *,
    n_interpolations: int,
    seed: int = 0,
    interp_mode: str = "linear",
) -> dict[str, ActivationStore]:
    """Augment activation stores with interpolated points.

    For each layer, sample random pairs (j, k) from the collected activations
    and create interpolated points:
        src_interp = (1 - t) * src[j] + t * src[k]
        tgt_interp = (1 - t) * tgt[j] + t * tgt[k]
    where t is sampled uniformly from (0, 1).  The interpolated pairs are fed
    back into the same ActivationStore so they contribute to the cross-covariance.

    interp_mode selects the path between the two points:
      "linear" -- straight line in activation space (default)
      "slerp"  -- great-circle in direction, linear in magnitude

    Source and target share the pair indices (j, k) and the coefficient t, which
    is what keeps the two new points in correspondence. Note that under slerp the
    resulting blend weights differ per side, because the angle between src[j] and
    src[k] is not the angle between tgt[j] and tgt[k].
    """
    if interp_mode not in {"linear", "slerp"}:
        raise ValueError(f"interp_mode must be 'linear' or 'slerp', got {interp_mode!r}")
    rng = torch.Generator().manual_seed(seed)

    for key, store in registry.items():
        src_rows, tgt_rows = store.rows(center=False)
        if src_rows is None or tgt_rows is None:
            continue
        n = src_rows.shape[0]
        if n < 2:
            continue

        # Sample random pairs and interpolation coefficients
        idx_j = torch.randint(0, n, (n_interpolations,), generator=rng)
        idx_k = torch.randint(0, n, (n_interpolations,), generator=rng)
        # Resample collisions so j != k
        collisions = idx_j == idx_k
        while collisions.any():
            idx_k[collisions] = torch.randint(0, n, (int(collisions.sum()),), generator=rng)
            collisions = idx_j == idx_k

        t = torch.rand(n_interpolations, 1, generator=rng)

        if interp_mode == "slerp":
            src_interp = _slerp_rows(src_rows[idx_j], src_rows[idx_k], t)
            tgt_interp = _slerp_rows(tgt_rows[idx_j], tgt_rows[idx_k], t)
        else:
            src_interp = (1.0 - t) * src_rows[idx_j] + t * src_rows[idx_k]
            tgt_interp = (1.0 - t) * tgt_rows[idx_j] + t * tgt_rows[idx_k]

        store.update(src_interp, tgt_interp)

    return registry


def _fps_anchor_indices(src: torch.Tensor, tgt: torch.Tensor, n_anchors: int) -> torch.Tensor:
    """Farthest-point sampling over the real rows, jointly on both sides.

    Each anchor contributes one probe function -- the geodesic field centred on
    it -- and the map is fit so those fields transport correctly. Two probes
    sitting next to each other are worth barely more than one, so what matters
    is that the set spans the cloud, not that any single point is unusual.

    Greedily add the row farthest from everything already chosen, scoring by the
    *smaller* of the two sides' distances: a row distinctive in the source but
    duplicated in the target is still an ambiguous anchor. MNIST's uniform
    background patches are duplicated on both sides, so that whole population
    collapses to a single pick instead of consuming most of the budget.

    Angular distance, matching the graph's similarity.
    """
    xs = torch.nn.functional.normalize(src.double(), dim=1)
    xt = torch.nn.functional.normalize(tgt.double(), dim=1)
    n = int(xs.shape[0])
    n_anchors = min(int(n_anchors), n)

    # Seed from the medoid, not an extreme point: FPS already leans toward
    # outliers and starting on one compounds it.
    first = int((xs @ xs.mean(dim=0, keepdim=True).T).squeeze(1).argmax())

    selected = [first]
    d_s = 1.0 - xs @ xs[first]
    d_t = 1.0 - xt @ xt[first]
    for _ in range(n_anchors - 1):
        score = torch.minimum(d_s, d_t)
        score[torch.tensor(selected, dtype=torch.long)] = -float("inf")
        j = int(torch.argmax(score))
        selected.append(j)
        d_s = torch.minimum(d_s, 1.0 - xs @ xs[j])
        d_t = torch.minimum(d_t, 1.0 - xt @ xt[j])

    return torch.tensor(sorted(selected), dtype=torch.long)


def _to_cpu_tensor(x, dtype=torch.float32) -> torch.Tensor | None:
    if x is None:
        return None
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().to(dtype)
    return torch.as_tensor(np.asarray(x), dtype=dtype)


def _cached_layer_transform(obj, expected: dict) -> tuple[torch.Tensor | None, str]:
    """Return a cached layer's T, or None with the reason it was rejected.

    The cache is addressed by directory, so nothing stops a run with different
    settings from loading maps built under the old ones -- silently, since a
    stale T is a perfectly valid tensor. Comparing the settings recorded
    alongside it turns that into an explicit recompute.
    """
    # An unverifiable cache is rejected rather than trusted. Accepting one
    # defeats the check exactly when it matters: layers saved before the
    # settings were recorded would be reused under any settings at all, so a
    # run overriding a knob would silently keep the maps built without it.
    if not isinstance(obj, dict):
        return None, "legacy format, settings unknown"
    saved = obj.get("settings")
    if not isinstance(saved, dict):
        return None, "no settings recorded"
    for field, want in expected.items():
        got = saved.get(field)
        if got != want:
            return None, f"{field} {got!r} != {want!r}"
    return obj.get("T"), "settings match"


def _fmap_payload_transform(obj) -> torch.Tensor:
    """Pull the transform out of a saved layer, old format or new.

    Layers used to be saved as the bare T tensor; they are now a dict holding
    the pieces the map is built from. Both must load.
    """
    if isinstance(obj, dict):
        return obj["T"]
    return obj


def _as_sorted_eigs(evals) -> np.ndarray:
    lam = np.asarray(
        evals.detach().cpu().numpy() if isinstance(evals, torch.Tensor) else evals,
        dtype=np.float64,
    ).ravel()
    return np.sort(lam)


def _local_gap_ratios(lam: np.ndarray, lo: int = 4) -> tuple[np.ndarray, np.ndarray]:
    """Spacings and their size relative to the local spacing scale.

    Returns ``(spacing, ratio)`` where ``spacing[i] = lam[i+1] - lam[i]`` and
    ``ratio[i]`` divides it by the median spacing in a multiplicative window
    around i. Spacings shrink like 1/k under Weyl, so a window proportional to
    k divides that trend out and leaves only genuine structure: a block
    boundary shows as a multiple of the local scale at any depth, while a
    featureless power-law spectrum sits near 1.
    """
    spacing = lam[1:] - lam[:-1]
    ratio = np.zeros_like(spacing)
    for i in range(lo, spacing.size):
        a = max(0, int(i / 1.6))
        b = min(spacing.size, int(i * 1.6) + 2)
        med = float(np.median(spacing[a:b]))
        ratio[i] = spacing[i] / med if med > 1e-15 else 0.0
    return spacing, ratio


def _choose_eig_cut(
    evals_src,
    evals_tgt,
    *,
    ceiling: int,
    k_min: int = 20,
    max_real: int | None = None,
    min_rel_gap: float = 1e-3,
    min_score: float = 2.0,
) -> tuple[int, str]:
    """Pick where to truncate the eigenbasis, from the spectrum itself.

    Scores each candidate by the *smaller* of the two sides' local gap ratios.
    ``min`` rather than a mean because C assumes the first k functions on one
    side correspond to the first k on the other: a cut that is clean on src and
    inside a degenerate block on tgt is still broken, and in practice the two
    sides disagree about where their structure lies.

    Among near-equal candidates the largest k wins -- a bigger basis is more
    expressive, and without that preference the choice drifts to small k where
    gaps are naturally wider.
    """
    lam_s, lam_t = _as_sorted_eigs(evals_src), _as_sorted_eigs(evals_tgt)
    hi = min(ceiling, lam_s.size - 1, lam_t.size - 1)
    if max_real is not None:
        # Past the real-sample count the basis describes interpolation paths
        # between the measured points rather than the points themselves.
        hi = min(hi, int(max_real) - 1)
    if hi <= k_min:
        return max(1, hi), f"range empty, using k={max(1, hi)}"

    _, r_s = _local_gap_ratios(lam_s)
    _, r_t = _local_gap_ratios(lam_t)
    rel_s = (lam_s[1:] - lam_s[:-1]) / np.maximum(np.abs(lam_s[:-1]), 1e-12)
    rel_t = (lam_t[1:] - lam_t[:-1]) / np.maximum(np.abs(lam_t[:-1]), 1e-12)

    scored: dict[int, float] = {}
    for k in range(k_min, hi + 1):
        i = k - 1                                  # gap crossed by keeping k
        if i >= r_s.size or i >= r_t.size:
            break
        # An absolute floor as well as a relative one: a cut between
        # eigenvalues agreeing to four decimals is pathological however it
        # scores against its neighbours.
        if rel_s[i] < min_rel_gap or rel_t[i] < min_rel_gap:
            continue
        scored[k] = float(min(r_s[i], r_t[i]))

    if scored:
        best_score = max(scored.values())
        # Largest k that is within 20% of the best: among equally clean cuts,
        # the more expressive basis wins.
        best_k = max(k for k, v in scored.items() if v >= 0.8 * best_score)
    else:
        best_k, best_score = -1, 0.0

    if best_k < 0 or best_score < min_score:
        # The floor applies here too. Rejecting a cutoff as degenerate and then
        # handing it back as the fallback is self-contradictory, and it lands on
        # exactly the cuts this rule exists to avoid: at the ceiling several
        # layer-sides sit at rel_gap ~ 1e-4. Walk down until both sides clear it.
        fallback = min(ceiling, hi)
        moved = 0
        while fallback > k_min:
            i = fallback - 1
            if i >= rel_s.size or i >= rel_t.size:
                fallback -= 1
                continue
            if rel_s[i] >= min_rel_gap and rel_t[i] >= min_rel_gap:
                break
            fallback -= 1
            moved += 1
        note = f" (walked down {moved} from the ceiling to clear {min_rel_gap:g})" if moved else ""
        return fallback, (
            f"no spectral structure (best {best_score:.2f}x < {min_score:.2f}x), "
            f"falling back to k={fallback}{note}"
        )
    return best_k, f"gap-selected k={best_k} ({best_score:.2f}x local, both sides)"


def _spectrum_report(evals, n_eig: int) -> str:
    """Describe the spectrum around a truncation at ``n_eig``.

    Eigenvector conditioning is governed by the gap to the neighbouring
    eigenvalue: inside a near-degenerate block only the subspace is determined,
    the individual vectors are an arbitrary rotation of it, so a cut landing
    inside such a block keeps a direction that carries no stable meaning. This
    reports the relative gap at the cut and the widest gap nearby, which is
    where a cut would land between blocks instead.
    """
    lam = _as_sorted_eigs(evals)
    if lam.size < 8:
        return "spectrum: too few eigenvalues"

    denom = np.maximum(np.abs(lam[:-1]), 1e-12)
    rel = (lam[1:] - lam[:-1]) / denom
    cut = min(max(n_eig - 1, 1), rel.size - 1)   # gap crossed by keeping n_eig

    lo = 4
    spacing, ratio = _local_gap_ratios(lam, lo=lo)
    best = int(np.argmax(ratio))

    near = "  ".join(f"{v:.4g}" for v in lam[max(0, n_eig - 3):n_eig + 3])
    return (
        f"spectrum: lam[{max(1, n_eig - 2)}..{min(lam.size, n_eig + 3)}]={near}  "
        f"rel_gap@{n_eig}={rel[cut]:.2e}  "
        f"cut_vs_local={ratio[cut]:.2f}x  "
        f"widest_in[{lo + 1},{spacing.size}]=k{best + 1}({ratio[best]:.2f}x local)"
    )


def _compute_fmap_from_activations(
    activation_registry: dict[str, ActivationStore],
    *,
    n_anchors_per_layer: dict[str, int] | None = None,
    n_anchors: int | None = None,
    anchor_select: str = "linspace",
    seed: int = 0,
    center: bool = False,
    num_eigs: int = 50,
    eig_select: str = "fixed",
    descr_weight_ref: int | None = 200,
    save_basis: bool = False,
    k_graph: int | None = None,
    device: str | torch.device = "cpu",
    verbose: bool = True,
    save_dir: str | None = None,
    precomputed: Mapping[str, torch.Tensor] | None = None,
) -> dict[str, torch.Tensor]:
    """Compute functional maps and return orthogonal T matrices per registry key.

    fmap.T has shape (d_src, d_tgt) — same convention as the Procrustes
    alignment maps used by ``_precompute_transforms``, so the returned
    tensors can directly replace them as ``t_in`` / ``t_out``.

    ``save_dir`` writes each layer's T to ``<save_dir>/<key>.pt`` as soon as it
    is computed, so a job that dies part-way keeps the layers it finished.
    ``precomputed`` holds maps recovered from a previous run; those layers are
    reused instead of recomputed.

    ``n_anchors`` caps how many index-correspondent points are used as landmarks
    for the descriptors. The cap drives cost: each anchor is one geodesic
    single-source shortest path per side, and one descriptor column.

    ``center`` subtracts the per-feature mean before the graphs are built. The
    graphs use angular similarity, so a large shared offset -- which transformer
    residual streams carry -- compresses every pairwise angle and flattens the
    neighbourhood structure. Only the correspondence estimate is centred; the
    Procrustes still fits the raw rows, so T keeps its meaning.
    """
    from .fmap_utils import FM_T
    from .fmap_utils.graph import build_graph

    dev = torch.device(device) if isinstance(device, str) else device
    log_prefix = "[theseus-fmap]"
    fmap_transforms: dict[str, torch.Tensor] = {}
    cached_layers = dict(precomputed or {})
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    for key, store in activation_registry.items():
        if key in cached_layers:
            expected = {
                "n_anchors": None if n_anchors is None else int(n_anchors),
                "anchor_select": str(anchor_select),
                "k_graph": None if k_graph is None else int(k_graph),
                "center": bool(center),
                "eig_select": str(eig_select),
                "num_eigs": int(num_eigs),
                "n_samples": int(store.n_samples),
                "descr_weight_ref": None if not descr_weight_ref else int(descr_weight_ref),
            }
            cached_T, why = _cached_layer_transform(cached_layers[key], expected)
            if cached_T is not None:
                fmap_transforms[key] = cached_T
                if verbose:
                    print(f"{log_prefix} {key}: reusing cached map T={tuple(cached_T.shape)} ({why})")
                store.h_a_list.clear()
                store.h_b_list.clear()
                continue
            if verbose:
                print(f"{log_prefix} {key}: cached map rejected ({why}) — recomputing")

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
        # Skip layers with too many points (e.g. raw pixel inputs) or tiny feature dim
        if n_samples > 100_000:
            if verbose:
                print(f"{log_prefix} {key}: skipped (too many points: {n_samples})")
            continue
        if d_src < 10 or d_tgt < 10:
            if verbose:
                print(f"{log_prefix} {key}: skipped (feature dim too small: {d_src}, {d_tgt})")
            continue

        # Anchors: the leading rows are the real samples, which share an index
        # correspondence between source and target (interpolated points are
        # appended after them).
        n_real = n_samples
        if n_anchors_per_layer is not None and key in n_anchors_per_layer:
            n_real = n_anchors_per_layer[key]
        n_real = min(n_real, n_samples)

        n_eig = min(num_eigs, n_samples - 1)
        k_eff = k_graph if k_graph is not None else max(int(n_samples * 0.07), 5)
        x_np = src_rows.numpy().astype(np.float64)
        y_np = tgt_rows.numpy().astype(np.float64)

        # Geometry the correspondence is estimated from; the Procrustes below
        # still fits the raw rows.
        if center:
            graph_src = (src_rows - src_rows.mean(dim=0, keepdim=True)).double()
            graph_tgt = (tgt_rows - tgt_rows.mean(dim=0, keepdim=True)).double()
        else:
            graph_src = torch.tensor(x_np, dtype=torch.float64)
            graph_tgt = torch.tensor(y_np, dtype=torch.float64)

        # Anchors are drawn only from the real rows: the interpolated points
        # carry no index correspondence between the two sides.
        if n_anchors is not None and 0 < int(n_anchors) < n_real:
            mode = str(anchor_select)
            if mode == "fps":
                anchor_idx = _fps_anchor_indices(
                    graph_src[:n_real], graph_tgt[:n_real], int(n_anchors)
                )
            elif mode == "random":
                gen = torch.Generator().manual_seed(int(seed))
                anchor_idx = torch.randperm(n_real, generator=gen)[: int(n_anchors)].sort().values
            else:
                # linspace: a lattice over the row index. Rows are image-major,
                # so it aliases against the token count -- which is why it is no
                # longer the only option.
                anchor_idx = torch.linspace(0, n_real - 1, int(n_anchors)).round().long().unique()
        else:
            anchor_idx = torch.arange(n_real)
        n_anch = int(anchor_idx.numel())
        anchors = torch.stack([anchor_idx, anchor_idx], dim=1)

        if verbose:
            print(
                f"{log_prefix} {key}: computing FM_T  "
                f"src={x_np.shape}  tgt={y_np.shape}  "
                f"anchors={n_anch}({anchor_select})  eigs={n_eig}  k={k_eff}  centered={bool(center)}"
            )

        try:
            # Build the two graphs up front so a degenerate one can be caught.
            # A disconnected kNN graph means duplicate or near-duplicate rows
            # (identical class-token embeddings, say) left nodes with no edge
            # under the distance kernel; the eigendecomposition that FM_T runs
            # next then dies with a hard segfault, taking the whole job with it.
            # Skipping the layer leaves it to the Procrustes fallback.
            # One kNN structure, two edge weightings, because the two consumers
            # want opposite things:
            #   - the Laplacian eigenbasis C is expressed in wants affinities
            #     (large = close), i.e. the gaussian kernel;
            #   - the dist_geod descriptors are shortest paths, so they want
            #     lengths (large = far), i.e. the distance kernel.
            # Weighting by distance throughout -- as this called for before --
            # hands the eigendecomposition the inverse of an affinity, so every
            # eigenvector, and C with them, comes out of a distorted basis.
            graphs = []
            disconnected = []
            connectivity = []
            for side, rows_t in (("src", graph_src), ("tgt", graph_tgt)):
                common = dict(algo="knn", similarity="angular", m=3, k=k_eff, device=dev)
                g_sim = build_graph(rows_t, kernel="gaussian", **common)
                g_dist = build_graph(rows_t, kernel="distance", **common)

                # Both graphs have to be connected. A disconnected affinity
                # graph segfaults the eigendecomposition; a disconnected
                # distance graph segfaults the geodesic shortest paths, which
                # is what conv1.out hits -- identical rows (MNIST's uniform
                # background patches) sit at distance 0, and a zero-weight edge
                # is dropped from the sparse matrix, isolating the node. The
                # gaussian kernel rescues the first case but not the second.
                sim_conn = bool(g_sim.G.isconnected())
                dist_conn = bool(g_dist.G.isconnected())
                connectivity.append(f"{side}=affinity:{sim_conn}/geodesic:{dist_conn}")
                if not sim_conn:
                    disconnected.append(f"{side}:affinity")
                if not dist_conn:
                    disconnected.append(f"{side}:geodesic")
                graphs.append((side, g_sim, g_dist))

            # FM prints this itself when it builds its own graphs; passing them
            # in skips that branch, so report it here instead.
            if verbose:
                print(f"{log_prefix} {key}: connected? {'  '.join(connectivity)}")

            if disconnected:
                print(
                    f"{log_prefix} {key}: skipped (disconnected graph: "
                    f"{', '.join(disconnected)}; k={k_eff}) — Procrustes fallback"
                )
                store.h_a_list.clear()
                store.h_b_list.clear()
                continue

            # Take the basis off the affinity graph, then point the object at
            # the distance-weighted graph so the geodesics run on lengths. FM
            # reads the basis from .eigvals/.eigvecs and the geodesics from .G,
            # so one object can carry both.
            # Decompose past the cut so the gap on the far side of it is
            # visible, then slice back to n_eig for the map itself. On the GPU
            # path the extra columns are free: eigh computes the whole spectrum
            # and the code merely slices it.
            n_log = min(int(n_eig * 1.5) + 2, n_samples - 1)
            decomposed = []
            for side, g_sim, g_dist in graphs:
                evals, evecs = g_sim.eigen_decomp(k=n_log)
                if verbose:
                    print(f"{log_prefix} {key}: {side} {_spectrum_report(evals, n_eig)}")
                decomposed.append((g_sim, g_dist, evals, evecs))

            # With "gap", n_eig is a ceiling and the cut comes from the
            # spectrum. Free to do here: the decomposition is already computed
            # and the graphs and descriptors do not depend on k at all, so only
            # the cheap tail of the pipeline sees the chosen value.
            k_use = n_eig
            if str(eig_select) == "gap":
                k_use, why = _choose_eig_cut(
                    decomposed[0][2], decomposed[1][2],
                    ceiling=n_eig, max_real=n_real,
                )
                if verbose:
                    print(f"{log_prefix} {key}: {why}")

            prepared = []
            for g_sim, g_dist, evals, evecs in decomposed:
                g_sim.eigvals, g_sim.eigvecs = evals[:k_use], evecs[:, :k_use]
                g_sim.G = g_dist.G
                prepared.append(g_sim)
            graphs = prepared

            # The energy is
            #   w_descr*||C A - B||^2 + w_dcomm*sum_i||C D_Ai - D_Bi C||^2
            #     + w_lap*||C L1 - L2 C||^2
            # A and B carry one column per anchor and the commutativity sum runs
            # over one operator per anchor, so both grow with the anchor count,
            # while the Laplacian term is normalised and does not. Left alone,
            # raising anchors quietly raises the descriptor weight and buries
            # w_lap (already 1e-3) -- changing the regularisation balance rather
            # than the information available. Scale the two anchor-dependent
            # weights by ref/p so the balance is the same at every anchor count,
            # with ref chosen so p = ref reproduces the defaults exactly.
            reg_weights = None
            if descr_weight_ref:
                scale = float(descr_weight_ref) / float(max(n_anch, 1))
                reg_weights = {
                    "w_descr": 1e0 * scale,
                    "w_dcomm": 1e-1 * scale,
                    "w_lap": 1e-3,
                    "w_orient": 0,
                }
                if verbose:
                    print(
                        f"{log_prefix} {key}: reg weights scaled x{scale:.3f} "
                        f"for {n_anch} anchors (ref {descr_weight_ref})"
                    )

            fmap = FM_T(
                torch.tensor(x_np, dtype=torch.float64),
                torch.tensor(y_np, dtype=torch.float64),
                anchors,
                transformation="orthogonal",
                num_eigs=k_use,
                reg_weights=reg_weights,
                graph_algo="knn",
                graph_similarity="angular",
                graph_kernel="distance",
                descriptors=("dist_geod",),
                k=k_eff,
                n_descr=1,
                compute_gt_map=False,
                refine=True,
                device=dev,
                graphs=tuple(graphs),
            )
            # fmap.T: (d_src, d_tgt) — same convention as Procrustes maps
            T = fmap.T
            if isinstance(T, torch.Tensor):
                fmap_transforms[key] = T.float().cpu()
            else:
                fmap_transforms[key] = torch.tensor(np.array(T), dtype=torch.float32)

            sim = fmap.get_similarity()
            c_shape = np.array(fmap.C).shape

            if save_dir:
                # Everything the map is made of, so a later question about it
                # does not mean rebuilding the graphs and geodesics: C, the
                # point-to-point map read out of it, the spectra, and the
                # settings that produced them. Phi in particular is what the
                # anchor hit-rate check needs. The eigenvectors are the only
                # bulky part -- (n x k) per side, tens of MB a layer -- so they
                # are opt-in; the rest is a few hundred KB.
                payload = {
                    "T": fmap_transforms[key],
                    "C": _to_cpu_tensor(fmap.C, torch.float32),
                    "Phi_flat": _to_cpu_tensor(getattr(fmap, "Phi_flat", None), torch.int64),
                    "eigvals_src": _to_cpu_tensor(graphs[0].eigvals, torch.float32),
                    "eigvals_tgt": _to_cpu_tensor(graphs[1].eigvals, torch.float32),
                    "anchors": anchor_idx.cpu().clone(),
                    "similarity": float(sim),
                    "settings": {
                        "n_anchors": None if n_anchors is None else int(n_anchors),
                        "anchor_select": str(anchor_select),
                        "k_graph": None if k_graph is None else int(k_graph),
                        "center": bool(center),
                        "eig_select": str(eig_select),
                        "num_eigs": int(num_eigs),
                        "n_samples": int(n_samples),
                        "descr_weight_ref": None if not descr_weight_ref else int(descr_weight_ref),
                    },
                    "n_eigs": int(k_use),
                    "n_anchors": int(n_anch),
                    "k_graph": int(k_eff),
                    "n_real": int(n_real),
                    "n_samples": int(n_samples),
                    "centered": bool(center),
                    "src_shape": tuple(x_np.shape),
                    "tgt_shape": tuple(y_np.shape),
                }
                if save_basis:
                    payload["eigvecs_src"] = _to_cpu_tensor(graphs[0].eigvecs, torch.float32)
                    payload["eigvecs_tgt"] = _to_cpu_tensor(graphs[1].eigvecs, torch.float32)
                torch.save(payload, os.path.join(save_dir, f"{key}.pt"))

            print(
                f"{log_prefix} {key}: fmap computed  "
                f"C={c_shape}  T={tuple(fmap_transforms[key].shape)}  "
                f"similarity={sim:.4f}"
                + (f"  saved -> {key}.pt" if save_dir else "")
            )
        except Exception as exc:
            print(f"{log_prefix} {key}: fmap computation failed — {exc}")

        # Free raw activations for this layer to avoid OOM
        store.h_a_list.clear()
        store.h_b_list.clear()

    return fmap_transforms


def _compute_procrustes_map_from_cov(cov: torch.Tensor) -> torch.Tensor:
    u, _, v_h = torch.linalg.svd(cov.double(), full_matrices=False)
    return (u @ v_h).float()


def _matrix_power_psd(matrix: torch.Tensor, *, power: float, eps: float) -> torch.Tensor:
    sym = 0.5 * (matrix + matrix.T)
    evals, evecs = torch.linalg.eigh(sym)
    powered = evals.clamp_min(float(eps)).pow(float(power))
    return (evecs * powered.unsqueeze(0)) @ evecs.T


def _partially_whiten_covariance(
    cov: torch.Tensor,
    *,
    a_gram: torch.Tensor,
    b_gram: torch.Tensor,
    power: float,
    eps: float,
) -> torch.Tensor:
    if power <= 0.0:
        return cov
    left = _matrix_power_psd(a_gram, power=-power, eps=eps)
    right = _matrix_power_psd(b_gram, power=-power, eps=eps)
    return left @ cov @ right


def _compute_alignment_map(
    store: ActivationStore,
    *,
    center: bool,
    whiten_power: float,
    whiten_eps: float,
) -> torch.Tensor | None:
    cov = store.get_covariance(center=center)
    if cov is None:
        return None
    if whiten_power > 0.0:
        a_gram = store.get_a_gram(center=center, epsilon=whiten_eps)
        b_gram = store.get_b_gram(center=center, epsilon=whiten_eps)
        if a_gram is not None and b_gram is not None:
            cov = _partially_whiten_covariance(
                cov,
                a_gram=a_gram,
                b_gram=b_gram,
                power=whiten_power,
                eps=whiten_eps,
            )
        else:
            logger.warning(
                "Theseus whitening requested but Gram statistics were unavailable; falling back to raw Procrustes."
            )
    return _compute_procrustes_map_from_cov(cov)


def _resolve_covariance_mode(mode: str) -> str:
    key = str(mode).strip().lower()
    if key in _ACTIVATION_COVARIANCE_MODES:
        return "activations"
    if key in _DATA_FREE_COVARIANCE_MODES:
        return "data_free"
    raise ValueError(
        "Theseus covariance_mode must be one of: activations, activation, data_free, data-free, weights, weight_space."
    )


def _compute_alignment_map_from_matrix_proxies(
    source_proxy: torch.Tensor,
    target_proxy: torch.Tensor,
    *,
    side: str,
    whiten_power: float,
    whiten_eps: float,
) -> torch.Tensor:
    source = source_proxy.detach().cpu().to(torch.float64)
    target = target_proxy.detach().cpu().to(torch.float64)

    if side == "input":
        a_gram = source.T @ source
        b_gram = target.T @ target
    elif side == "output":
        a_gram = source @ source.T
        b_gram = target @ target.T
    else:  # pragma: no cover - defensive
        raise ValueError(f"Unsupported alignment side '{side}'.")

    u_a, s_a, _ = torch.linalg.svd(a_gram, full_matrices=False)
    u_b, s_b, _ = torch.linalg.svd(b_gram, full_matrices=False)
    rank = min(int(u_a.shape[1]), int(u_b.shape[1]))
    basis_power = 0.5 - float(whiten_power)
    scale_a = s_a[:rank].clamp_min(float(whiten_eps)).pow(basis_power)
    scale_b = s_b[:rank].clamp_min(float(whiten_eps)).pow(basis_power)
    source_basis = u_a[:, :rank] * scale_a.unsqueeze(0)
    target_basis = u_b[:, :rank] * scale_b.unsqueeze(0)
    cov = source_basis @ target_basis.T

    return _compute_procrustes_map_from_cov(cov)


def _transport_weight(delta_weight: torch.Tensor, t_in: torch.Tensor, t_out: torch.Tensor, *, key: str) -> torch.Tensor:
    if key == "proj":
        return (t_out.T @ delta_weight.T @ t_in).T
    return t_out.T @ delta_weight @ t_in


def _transport_bias(delta_vec: torch.Tensor, t_out: torch.Tensor) -> torch.Tensor:
    return delta_vec @ t_out


def _param_to_module(visual_model: torch.nn.Module) -> dict[str, str]:
    out: dict[str, str] = {}
    for module_name, module in visual_model.named_modules():
        for param_name, _ in module.named_parameters(recurse=False):
            full_name = f"{module_name}.{param_name}" if module_name else param_name
            out[full_name] = module_name
    return out


def _iter_with_progress(iterable: Any, *, total: int, desc: str, enabled: bool) -> Any:
    if not enabled or tqdm is None:
        return iterable
    return tqdm(iterable, total=total, desc=desc, leave=False)


def _stratified_perm(
    dataset: Any,
    n_samples: int,
    n_batches: int | None,
    batch_size: int,
    generator: torch.Generator,
) -> torch.Tensor | None:
    """Pick b*s/n_classes distinct samples per class, shuffled."""
    from collections import defaultdict

    # Get labels — try .targets first (torchvision), then HF column
    targets = getattr(dataset, "targets", None)
    if targets is None:
        targets = getattr(dataset, "label", None)
    if targets is None:
        return None

    if hasattr(targets, "tolist"):
        labels = targets.tolist()[:n_samples]
    elif isinstance(targets, list):
        labels = targets[:n_samples]
    else:
        labels = [int(targets[i]) for i in range(n_samples)]

    # Group indices by class
    class_indices: dict[int, list[int]] = defaultdict(list)
    for i, lbl in enumerate(labels):
        class_indices[lbl].append(i)

    n_classes = len(class_indices)
    if n_classes < 2:
        return None

    total = n_samples if n_batches is None else min(n_samples, n_batches * batch_size)
    per_class = max(1, total // n_classes)

    selected: list[int] = []
    for cls in sorted(class_indices.keys()):
        pool = torch.tensor(class_indices[cls])
        pool = pool[torch.randperm(len(pool), generator=generator)]
        selected.extend(pool[:per_class].tolist())

    perm = torch.tensor(selected)
    perm = perm[torch.randperm(len(perm), generator=generator)]

    print(f"[theseus] Balanced sampling: {len(perm)} samples, {per_class}/class, {n_classes} classes")
    return perm


def _iter_random_dataset_batches(
    source_dataloader: Iterable[Any],
    target_dataloader: Iterable[Any],
    *,
    n_batches: int | None,
    seed: int,
    batch_size: int | None,
) -> Iterable[tuple[Any, Any]] | None:
    source_dataset = getattr(source_dataloader, "dataset", None)
    target_dataset = getattr(target_dataloader, "dataset", None)
    if source_dataset is None or target_dataset is None:
        return None

    try:
        n_source = int(len(source_dataset))
        n_target = int(len(target_dataset))
    except Exception:
        return None

    n_samples = min(n_source, n_target)
    if n_samples <= 0:
        return iter(())

    if batch_size is None:
        source_bs = getattr(source_dataloader, "batch_size", None)
        target_bs = getattr(target_dataloader, "batch_size", None)
        if source_bs is None or target_bs is None:
            return None
        batch_size = min(int(source_bs), int(target_bs))
    else:
        batch_size = int(batch_size)

    if batch_size <= 0:
        return None

    source_collate = getattr(source_dataloader, "collate_fn", None)
    target_collate = getattr(target_dataloader, "collate_fn", None)
    if not callable(source_collate) or not callable(target_collate):
        return None

    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))

    # Try stratified sampling: equal samples per class
    perm = _stratified_perm(source_dataset, n_samples, n_batches, batch_size, generator)
    if perm is None:
        # Fallback to random
        perm = torch.randperm(n_samples, generator=generator)
        if n_batches is not None:
            max_items = min(n_samples, int(n_batches) * batch_size)
            perm = perm[:max_items]

    def _iterator() -> Iterable[tuple[Any, Any]]:
        for start in range(0, int(perm.numel()), batch_size):
            indices = perm[start : start + batch_size].tolist()
            source_items = [source_dataset[i] for i in indices]
            target_items = [target_dataset[i] for i in indices]
            yield source_collate(source_items), target_collate(target_items)

    return _iterator()


@dataclass(frozen=True)
class _LayerTransform:
    kind: str
    t_in: torch.Tensor | None = None
    t_out: torch.Tensor | None = None


def _precompute_transforms(
    *,
    target_model: torch.nn.Module,
    target_visual_base: Mapping[str, torch.Tensor],
    visual_delta: Mapping[str, torch.Tensor],
    activation_registry: Mapping[str, ActivationStore],
    center_acts: bool,
    whiten_power: float,
    whiten_eps: float,
    show_progress: bool,
    method_name: str,
    fmap_transforms: Mapping[str, torch.Tensor] | None = None,
) -> dict[str, _LayerTransform]:
    transforms_by_key: dict[str, _LayerTransform] = {}
    t_out_cache: dict[str, torch.Tensor] = {}
    visual_model = _visual_module(target_model)
    param_to_module = _param_to_module(visual_model)
    use_fmap = bool(fmap_transforms)

    items = _iter_with_progress(
        visual_delta.items(),
        total=len(visual_delta),
        desc=f"{method_name}.prepare: compute transforms",
        enabled=show_progress,
    )
    for key, delta_source in items:
        if key not in target_visual_base:
            print(f"Theseus align: skipping task vector key:{key} as it is not in target visual base.")
            continue

        if key in _ZERO_KEYS:
            transforms_by_key[key] = _LayerTransform(kind="zero")
            continue

        module_name = param_to_module.get(key, key.rsplit(".", 1)[0] if "." in key else "")
        if key == "proj":
            in_key = "ln_post.out"
            out_key = ".out"
        else:
            in_key = f"{module_name}.in"
            out_key = f"{module_name}.out"

        if delta_source.ndim == 2:
            # Try fmap transforms first, fall back to Procrustes
            t_in: torch.Tensor | None = None
            t_out: torch.Tensor | None = None

            if use_fmap and in_key in fmap_transforms:
                t_in = fmap_transforms[in_key]
            if use_fmap and out_key in fmap_transforms:
                t_out = fmap_transforms[out_key]

            # Fall back to Procrustes for any missing fmap transform
            if t_in is None or t_out is None:
                in_store = activation_registry.get(in_key)
                out_store = activation_registry.get(out_key)
                if in_store is not None and out_store is not None:
                    if t_in is None:
                        t_in = _compute_alignment_map(
                            in_store,
                            center=center_acts,
                            whiten_power=whiten_power,
                            whiten_eps=whiten_eps,
                        )
                    if t_out is None:
                        t_out = _compute_alignment_map(
                            out_store,
                            center=center_acts,
                            whiten_power=whiten_power,
                            whiten_eps=whiten_eps,
                        )

            if t_in is not None and t_out is not None:
                transforms_by_key[key] = _LayerTransform(kind="weight", t_in=t_in, t_out=t_out)
                continue
            transforms_by_key[key] = _LayerTransform(kind="weight")
            continue

        if delta_source.ndim == 1:
            if key.endswith(".bias"):
                weight_key = f"{key[: -len('.bias')]}.weight"
                weight_transform = transforms_by_key.get(weight_key)
                if weight_transform is not None and weight_transform.t_out is not None:
                    transforms_by_key[key] = _LayerTransform(kind="bias", t_out=weight_transform.t_out)
                    continue

            # Robustness fallback: covers uncommon ordering/edge cases where
            # the bias has no directly available weight transform yet.
            cached_t_out = t_out_cache.get(out_key)
            if cached_t_out is not None:
                transforms_by_key[key] = _LayerTransform(kind="bias", t_out=cached_t_out)
                continue

            # Try fmap transform for bias
            if use_fmap and out_key in fmap_transforms:
                t_out_cache[out_key] = fmap_transforms[out_key]
                transforms_by_key[key] = _LayerTransform(kind="bias", t_out=fmap_transforms[out_key])
                continue

            out_store = activation_registry.get(out_key)
            if out_store is not None:
                t_out = _compute_alignment_map(
                    out_store,
                    center=center_acts,
                    whiten_power=whiten_power,
                    whiten_eps=whiten_eps,
                )
                if t_out is not None:
                    t_out_cache[out_key] = t_out
                    transforms_by_key[key] = _LayerTransform(kind="bias", t_out=t_out)
                    continue
            transforms_by_key[key] = _LayerTransform(kind="bias")
            continue

        transforms_by_key[key] = _LayerTransform(kind="unsupported")

    return transforms_by_key


def _precompute_transforms_data_free(
    *,
    source_visual_base: Mapping[str, torch.Tensor],
    target_visual_base: Mapping[str, torch.Tensor],
    visual_delta: Mapping[str, torch.Tensor],
    whiten_power: float,
    whiten_eps: float,
    show_progress: bool,
    method_name: str,
) -> dict[str, _LayerTransform]:
    transforms_by_key: dict[str, _LayerTransform] = {}

    items = _iter_with_progress(
        visual_delta.items(),
        total=len(visual_delta),
        desc=f"{method_name}.prepare: compute data-free transforms",
        enabled=show_progress,
    )
    for key, delta_source in items:
        if key not in target_visual_base:
            print(f"Theseus align: skipping task vector key:{key} as it is not in target visual base.")
            continue
        if key not in source_visual_base:
            transforms_by_key[key] = _LayerTransform(kind="unsupported")
            continue

        source_ref = source_visual_base[key]
        target_ref = target_visual_base[key]

        if key in _ZERO_KEYS:
            transforms_by_key[key] = _LayerTransform(kind="zero")
            continue

        if delta_source.ndim == 2 and source_ref.ndim == 2 and target_ref.ndim == 2:
            # CLIP stores `visual.proj` as [hidden, embed], so its axes are
            # flipped relative to nn.Linear([out, in]) weights.
            t_in = _compute_alignment_map_from_matrix_proxies(
                source_ref,
                target_ref,
                side="output" if key == "proj" else "input",
                whiten_power=whiten_power,
                whiten_eps=whiten_eps,
            )
            t_out = _compute_alignment_map_from_matrix_proxies(
                source_ref,
                target_ref,
                side="input" if key == "proj" else "output",
                whiten_power=whiten_power,
                whiten_eps=whiten_eps,
            )
            transforms_by_key[key] = _LayerTransform(kind="weight", t_in=t_in, t_out=t_out)
            continue

        if delta_source.ndim == 1 and source_ref.ndim == 1 and target_ref.ndim == 1:
            if key.endswith(".bias"):
                weight_key = f"{key[: -len('.bias')]}.weight"
                weight_transform = transforms_by_key.get(weight_key)
                if weight_transform is not None and weight_transform.t_out is not None:
                    transforms_by_key[key] = _LayerTransform(kind="bias", t_out=weight_transform.t_out)
                    continue

            t_out = _compute_alignment_map_from_matrix_proxies(
                source_ref.unsqueeze(0),
                target_ref.unsqueeze(0),
                side="input",
                whiten_power=whiten_power,
                whiten_eps=whiten_eps,
            )
            transforms_by_key[key] = _LayerTransform(kind="bias", t_out=t_out)
            continue

        transforms_by_key[key] = _LayerTransform(kind="unsupported")

    return transforms_by_key


def _apply_transforms_to_visual_delta(
    *,
    target_visual_base: Mapping[str, torch.Tensor],
    visual_delta: Mapping[str, torch.Tensor],
    transforms_by_key: Mapping[str, _LayerTransform],
    show_progress: bool,
    method_name: str,
) -> TensorDict:
    aligned: TensorDict = {}

    items = _iter_with_progress(
        visual_delta.items(),
        total=len(visual_delta),
        desc=f"{method_name}.apply: transport params",
        enabled=show_progress,
    )
    for key, delta_source in items:
        if key not in target_visual_base:
            print(f"Theseus align: skipping task vector key:{key} as it is not in target visual base.")
            continue

        target_ref = target_visual_base[key]
        transported = torch.zeros_like(target_ref, dtype=torch.float32, device="cpu")

        transform = transforms_by_key.get(key)
        if transform is not None:
            if (
                transform.kind == "weight"
                and delta_source.ndim == 2
                and transform.t_in is not None
                and transform.t_out is not None
            ):
                try:
                    transported = _transport_weight(
                        delta_source.float().cpu(), transform.t_in, transform.t_out, key=key
                    )
                except RuntimeError as exc:
                    logger.warning("Theseus transport failed for %s: %s", key, exc)
            elif transform.kind == "bias" and delta_source.ndim == 1 and transform.t_out is not None:
                try:
                    transported = _transport_bias(delta_source.float().cpu(), transform.t_out)
                except ValueError as exc:
                    logger.warning("Theseus vector transport failed for %s: %s", key, exc)

        if transported.shape != target_ref.shape:
            logger.warning(
                "Theseus produced wrong shape for %s: got %s expected %s. Zeroing.",
                key,
                tuple(transported.shape),
                tuple(target_ref.shape),
            )
            transported = torch.zeros_like(target_ref, dtype=torch.float32, device="cpu")

        aligned[key] = transported.to(dtype=target_ref.dtype, device=target_ref.device)

    return aligned


@dataclass(frozen=True)
class TheseusRebase:
    """Transport matrix task-vector updates with activation-aligned layer maps.

    ``prepare`` estimates source-to-target coordinate transforms from activations
    or data-free covariance information. ``apply`` reuses those transforms for
    a compatible delta, making alpha sweeps cheaper than rebuilding alignment.
    """

    name: str = "theseus"

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
        covariance_mode: str = "activations",
        whiten_power: float = 0.0,
        whiten_eps: float = 1e-6,
        n_batches: int | None = None,
        num_batches: int | None = None,
        seed: int = 0,
        batch_size: int | None = None,
        patch_qkv: bool = True,
        n_interpolations: int = 0,
        interp_mode: str = "linear",
        use_fmap: bool = False,
        fmap_num_eigs: int = 50,
        fmap_k_graph: int | None = None,
        fmap_n_anchors: int | None = None,
        fmap_anchor_select: str = "linspace",
        fmap_eig_select: str = "fixed",
        fmap_descr_weight_ref: int | None = 200,
        fmap_save_basis: bool = False,
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
        use_fmap = bool(use_fmap)
        # Config fallbacks num_batches -> n_batches
        if n_batches is None:
            n_batches = num_batches
        log_prefix = f"[{self.name}]"
        covariance_mode = _resolve_covariance_mode(covariance_mode)
        whiten_power = float(whiten_power)
        whiten_eps = float(whiten_eps)
        if not (0.0 <= whiten_power <= 0.5):
            raise ValueError("Theseus whiten_power must be in [0, 0.5].")
        if whiten_eps <= 0.0:
            raise ValueError("Theseus whiten_eps must be > 0.")

        if verbose:
            print(
                f"{log_prefix} prepare: start "
                f"(seq_align={seq_align}, center_acts={bool(center_acts)}, "
                f"whiten_power={whiten_power}, covariance_mode={covariance_mode}, "
                f"n_batches={n_batches}, seed={int(seed)})"
            )

        # Resolve which models to use for activation extraction
        act_source_model = activation_source_model if activation_source_model is not None else source_model
        act_target_model = activation_target_model if activation_target_model is not None else target_model
        using_separate_act_models = (act_source_model is not source_model) or (act_target_model is not target_model)
        if verbose and using_separate_act_models:
            print(f"{log_prefix} prepare: using separate models for activation extraction")

        patched_source = 0
        patched_target = 0
        if patch_qkv:
            if verbose:
                print(f"{log_prefix} prepare: patching fused qkv blocks if needed")
            patched_source = _split_fused_qkv_if_needed(act_source_model)
            patched_target = _split_fused_qkv_if_needed(act_target_model)
            if patched_source > 0 or patched_target > 0:
                logger.info(
                    "%s prepare: split fused qkv attention blocks (source=%d, target=%d)",
                    self.name,
                    patched_source,
                    patched_target,
                )
        elif verbose:
            print(f"{log_prefix} prepare: patch_qkv disabled")

        activation_registry: dict[str, ActivationStore] = {}
        transforms_by_key: dict[str, _LayerTransform] = {}
        fmap_transforms: dict[str, torch.Tensor] = {}
        # Determine whether rebase state dicts need QKV splitting.
        # When using separate activation models, the rebase models were not
        # patched, so check them independently for fused MHA blocks.
        if using_separate_act_models:
            rebase_has_fused = patch_qkv and (
                _has_fused_mha(_visual_module(source_model))
                or _has_fused_mha(_visual_module(target_model))
            )
        else:
            rebase_has_fused = False
        split_fused_qkv = bool(
            patch_qkv and (patched_source > 0 or patched_target > 0 or rebase_has_fused)
        )
        unpatched_source = 0
        unpatched_target = 0
        try:
            if covariance_mode == "activations":
                _act_path = activations_path if activations_path else None

                if _act_path and os.path.isfile(_act_path):
                    # Load cached activations
                    activation_registry, n_real_samples_per_layer = _load_activation_registry(_act_path)
                    if verbose:
                        print(f"{log_prefix} prepare: loaded cached activations ({len(activation_registry)} layers)")
                else:
                    if source_dataloader is None or target_dataloader is None:
                        raise ValueError(
                            "Theseus activation covariance mode requires both source_dataloader and target_dataloader."
                        )
                    if verbose:
                        print(f"{log_prefix} prepare: collecting activations")

                    activation_registry = collect_activations(
                        act_source_model,
                        act_target_model,
                        source_dataloader,
                        target_dataloader,
                        device=device,
                        seq_align=seq_align,
                        n_batches=n_batches,
                        seed=int(seed),
                        batch_size=batch_size,
                        store_raw=n_interpolations > 0 or use_fmap,
                        store_a_gram=whiten_power > 0.0,
                        store_b_gram=whiten_power > 0.0,
                    )
                    if verbose:
                        print(f"{log_prefix} prepare: collected activation entries = {len(activation_registry)}")

                    # Save per-layer sample counts (real correspondences) before interpolation
                    n_real_samples_per_layer = {
                        key: store.n_samples for key, store in activation_registry.items()
                    }

                    if n_interpolations > 0:
                        activation_registry = _augment_registry_with_interpolations(
                            activation_registry,
                            n_interpolations=n_interpolations,
                            seed=int(seed),
                            interp_mode=str(interp_mode),
                        )
                        if verbose:
                            sample_store = next(iter(activation_registry.values()), None)
                            n_total = sample_store.n_samples if sample_store else 0
                            print(f"{log_prefix} prepare: augmented with {n_interpolations} {interp_mode} interpolations (total samples per layer: {n_total})")

                    # Save to disk for reuse
                    if _act_path:
                        _save_activation_registry(activation_registry, _act_path, n_real_samples_per_layer)

                if use_fmap:
                    _fmap_path = fmap_transforms_path if fmap_transforms_path else None
                    if _fmap_path and os.path.isfile(_fmap_path):
                        fmap_transforms = {
                            k: _fmap_payload_transform(v)
                            for k, v in torch.load(
                                _fmap_path, map_location="cpu", weights_only=False
                            ).items()
                        }
                        if verbose:
                            print(f"{log_prefix} prepare: loaded precomputed fmap transforms ({len(fmap_transforms)} layers) from {_fmap_path}")
                    else:
                        # A directory may hold a complete cache or the layers a
                        # previous run finished before dying; either way the
                        # missing layers are recomputed below.
                        cached: dict[str, torch.Tensor] = {}
                        if _fmap_path and os.path.isdir(_fmap_path):
                            for fname in os.listdir(_fmap_path):
                                if fname.endswith(".pt"):
                                    layer_key = fname[:-3]  # strip .pt
                                    # raw payload: the settings recorded in it
                                    # are checked before the map is reused
                                    cached[layer_key] = torch.load(
                                        os.path.join(_fmap_path, fname),
                                        map_location="cpu", weights_only=False,
                                    )
                            if verbose:
                                print(f"{log_prefix} prepare: loaded {len(cached)} cached fmap transforms from {_fmap_path}")
                        n_missing = sum(1 for key in activation_registry if key not in cached)
                        if verbose:
                            print(f"{log_prefix} prepare: computing functional maps ({n_missing} layers to go)")
                        fmap_transforms = _compute_fmap_from_activations(
                            activation_registry,
                            n_anchors_per_layer=n_real_samples_per_layer,
                            n_anchors=int(fmap_n_anchors) if fmap_n_anchors else None,
                            anchor_select=str(fmap_anchor_select or "linspace"),
                            seed=int(seed),
                            center=bool(center_acts),
                            num_eigs=int(fmap_num_eigs),
                            eig_select=str(fmap_eig_select),
                            descr_weight_ref=fmap_descr_weight_ref,
                            save_basis=bool(fmap_save_basis),
                            k_graph=fmap_k_graph,
                            device=device,
                            verbose=bool(verbose),
                            save_dir=_fmap_path,
                            precomputed=cached,
                        )
                        if _fmap_path and verbose:
                            print(f"{log_prefix} prepare: fmap transforms saved under {_fmap_path}")
                    if verbose:
                        print(f"{log_prefix} prepare: fmap transforms for {len(fmap_transforms)} layers")

            elif verbose:
                print(f"{log_prefix} prepare: skipping activation collection (data-free covariance mode)")

            if target_base is not None and delta is not None:
                if verbose:
                    print(f"{log_prefix} prepare: precomputing per-layer transforms")
                visual_key_map = _visual_delta_keys(delta)
                source_visual_base = _visual_state_dict(source_model.state_dict())
                target_visual_base = _visual_state_dict(target_base)
                visual_delta = {
                    stripped_key: delta[original_key]
                    for stripped_key, original_key in visual_key_map.items()
                    if stripped_key in target_visual_base
                }

                if split_fused_qkv:
                    source_visual_base = _split_fused_qkv_state(source_visual_base)
                    target_visual_base = _split_fused_qkv_state(target_visual_base)
                    visual_delta = _split_fused_qkv_state(visual_delta)

                if covariance_mode == "activations":
                    transforms_by_key = _precompute_transforms(
                        target_model=target_model,
                        target_visual_base=target_visual_base,
                        visual_delta=visual_delta,
                        activation_registry=activation_registry,
                        center_acts=bool(center_acts),
                        whiten_power=whiten_power,
                        whiten_eps=whiten_eps,
                        show_progress=bool(show_progress),
                        method_name=self.name,
                        fmap_transforms=fmap_transforms if fmap_transforms else None,
                    )
                else:
                    transforms_by_key = _precompute_transforms_data_free(
                        source_visual_base=source_visual_base,
                        target_visual_base=target_visual_base,
                        visual_delta=visual_delta,
                        whiten_power=whiten_power,
                        whiten_eps=whiten_eps,
                        show_progress=bool(show_progress),
                        method_name=self.name,
                    )
                if verbose:
                    print(f"{log_prefix} prepare: computed transforms = {len(transforms_by_key)}")
            elif verbose:
                print(f"{log_prefix} prepare: target_base/delta missing, skipping transform precompute")
        finally:
            if patch_qkv and (patched_source > 0 or patched_target > 0):
                try:
                    unpatched_source = int(merge_openclip_vit_attn(_visual_module(act_source_model)))
                    unpatched_target = int(merge_openclip_vit_attn(_visual_module(act_target_model)))
                    if verbose:
                        print(
                            f"{log_prefix} prepare: recomposed fused qkv blocks "
                            f"(source={unpatched_source}, target={unpatched_target})"
                        )
                except Exception as exc:
                    logger.warning("%s prepare: failed to recompose patched attention blocks: %s", self.name, exc)

        if verbose:
            print(f"{log_prefix} prepare: done")

        return {
            "activation_registry": activation_registry,
            "transforms_by_key": transforms_by_key,
            "covariance_mode": covariance_mode,
            "split_fused_qkv": split_fused_qkv,
            "n_batches": n_batches,
            "patched_source_blocks": patched_source,
            "patched_target_blocks": patched_target,
            "unpatched_source_blocks": unpatched_source,
            "unpatched_target_blocks": unpatched_target,
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

        if verbose:
            print(f"{log_prefix} apply: start")

        transforms_by_key = prepared.get("transforms_by_key", None)
        if transforms_by_key is None:
            raise ValueError("Theseus prepared payload is missing 'transforms_by_key'.")

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

        if strict and not visual_delta_work:
            raise ValueError("Theseus did not find any visual delta keys to transport.")

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
                out[original_key] = torch.zeros_like(target_base[original_key], device=target_base[original_key].device)
            processed.add(original_key)

        for key in delta:
            if key in processed or key not in target_base:
                continue
            out[key] = torch.zeros_like(target_base[key], device=target_base[key].device)

        if strict:
            missing = sorted(set(delta.keys()) - set(out.keys()))
            if missing:
                raise KeyError(f"Theseus did not transport all delta keys. Example: {missing[:10]}")

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
        covariance_mode: str = "activations",
        whiten_power: float = 0.0,
        whiten_eps: float = 1e-6,
        prepared: Mapping[str, Any] | None = None,
        n_batches: int | None = None,
        num_batches: int | None = None,
        seed: int = 0,
        batch_size: int | None = None,
        patch_qkv: bool = True,
        n_interpolations: int = 0,
        interp_mode: str = "linear",
        use_fmap: bool = False,
        fmap_num_eigs: int = 50,
        fmap_k_graph: int | None = None,
        fmap_n_anchors: int | None = None,
        fmap_anchor_select: str = "linspace",
        fmap_eig_select: str = "fixed",
        fmap_descr_weight_ref: int | None = 200,
        fmap_save_basis: bool = False,
        activations_path: str | None = None,
        fmap_transforms_path: str | None = None,
        verbose: bool = True,
        show_progress: bool = True,
        **kwargs,
    ) -> TensorDict:
        del source_base
        log_prefix = f"[{self.name}]"

        if n_batches is None:
            n_batches = num_batches

        prepared_payload: Mapping[str, Any]
        if prepared is None:
            if source_model is None or target_model is None:
                raise ValueError("Theseus transport requires both source_model and target_model.")
            if _resolve_covariance_mode(covariance_mode) == "activations" and (
                source_dataloader is None or target_dataloader is None
            ):
                raise ValueError("Theseus transport requires both source_dataloader and target_dataloader.")

            prepared_payload = self.prepare(
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
                covariance_mode=covariance_mode,
                whiten_power=float(whiten_power),
                whiten_eps=float(whiten_eps),
                n_batches=n_batches,
                seed=int(seed),
                batch_size=batch_size,
                patch_qkv=patch_qkv,
                n_interpolations=int(n_interpolations),
                interp_mode=str(interp_mode),
                use_fmap=bool(use_fmap),
                fmap_num_eigs=int(fmap_num_eigs),
                fmap_k_graph=fmap_k_graph,
                fmap_n_anchors=fmap_n_anchors,
                fmap_anchor_select=str(fmap_anchor_select or "linspace"),
                fmap_eig_select=str(fmap_eig_select),
                fmap_descr_weight_ref=fmap_descr_weight_ref,
                fmap_save_basis=bool(fmap_save_basis),
                activations_path=activations_path,
                fmap_transforms_path=fmap_transforms_path,
                verbose=bool(verbose),
                show_progress=bool(show_progress),
                **kwargs,
            )
        else:
            prepared_payload = prepared
            if verbose:
                print(f"{log_prefix} transport: using provided prepared payload")

        return self.apply(
            prepared_payload,
            target_base=target_base,
            delta=delta,
            strict=bool(strict),
            verbose=bool(verbose),
            show_progress=bool(show_progress),
        )


register(TheseusRebase())
