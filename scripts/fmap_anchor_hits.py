"""Do the functional maps recover the correspondences they were given?

The anchors are the one place where ground truth exists: row i of the source
activations and row i of the target come from the same input, so a map that
works should send anchor i back to anchor i. Phi is saved alongside the
anchors, so this needs no recomputation.

    python scripts/fmap_anchor_hits.py <fmap_dir> [--images 32]

Two rates are reported. "exact" is Phi[i] == i, which is harsh: tokens within
one image are near-identical, so landing on a neighbouring token of the right
image is close to right. "same-image" allows that, and is the one to read when
exact is small. Both are printed next to the rate a random map would achieve.
"""
import argparse
import os

import torch


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("fmap_dir")
    p.add_argument("--images", type=int, default=32,
                   help="images in the calibration batch (num_batches * batch_size)")
    p.add_argument("--limit", type=int, default=None, help="only the first N layers")
    args = p.parse_args()

    files = sorted(f for f in os.listdir(args.fmap_dir) if f.endswith(".pt"))
    if args.limit:
        files = files[: args.limit]

    print(f"{'layer':<52} {'exact':>9} {'chance':>9} {'same-img':>9} {'chance':>9} {'k':>5}")
    tot_exact = tot_same = tot_n = 0
    for fname in files:
        d = torch.load(os.path.join(args.fmap_dir, fname), map_location="cpu", weights_only=False)
        if not isinstance(d, dict) or "Phi_flat" not in d or d["Phi_flat"] is None:
            print(f"{fname[:-3]:<52} {'no Phi saved (old format)':>40}")
            continue

        phi, anch = d["Phi_flat"], d["anchors"]
        n_samples, n_real = int(d["n_samples"]), int(d["n_real"])
        # rows are image-major: row = image * tokens_per_image + token
        tpi = max(1, n_real // max(1, args.images))

        hit = phi[anch]
        exact = (hit == anch).float().mean().item()
        same = ((hit // tpi) == (anch // tpi)).float().mean().item()
        # a uniformly random map lands on the right row 1/n of the time, and
        # somewhere in the right image tpi/n of the time
        c_exact, c_same = 1.0 / n_samples, tpi / n_samples

        print(f"{fname[:-3]:<52} {exact:>9.4f} {c_exact:>9.5f} {same:>9.4f} {c_same:>9.5f} {int(d['n_eigs']):>5}")
        tot_exact += exact * len(anch)
        tot_same += same * len(anch)
        tot_n += len(anch)

    if tot_n:
        print(f"\n{'WEIGHTED MEAN':<52} {tot_exact / tot_n:>9.4f} {'':>9} {tot_same / tot_n:>9.4f}")


if __name__ == "__main__":
    main()
