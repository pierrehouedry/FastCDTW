"""
1-NN classification on the UCR archive
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from checkpoint import read_locked, update_json
from data import cv_splits, dataset_names, load_ucr_split
from distances import (METHODS, cdist_dtw, cdist_euclidean, cdist_softdtw,
                       fastcdtw_cdist, one_nn_accuracy)

GAMMAS = [10.0 ** k for k in range(-4, 5)]

RESULTS_DIR = HERE / "results"
CKPT_DIR = RESULTS_DIR / "checkpoints"
LOG_DIR = RESULTS_DIR / "logs"
CHECKPOINT_PATH = CKPT_DIR / "ucr_nn_checkpoint.json"
SELECTION_PATH = RESULTS_DIR / "ucr_nn_datasets.csv"


def cross_validate(values, score_one, name: str, log=print) -> tuple[float, dict]:
    """Pick the value with the best mean held-out 1-NN accuracy."""
    scores = {}
    for v in values:
        accs = [a for a in score_one(v) if np.isfinite(a)]
        scores[v] = float(np.mean(accs)) if accs else float("nan")
        log(f"      {name}={v:<10g} held-out acc {scores[v]:.3f}")
    best = max(scores, key=lambda v: (scores[v] if np.isfinite(scores[v]) else -1))
    log(f"    {name} CV: best={best:g} (held-out acc {scores[best]:.3f})")
    return best, scores


def select_gamma(X_train, y_train, args, log=print) -> tuple[float, dict]:
    splits = cv_splits(len(X_train), args.gamma_splits, args.seed)

    def score_one(g):
        return [one_nn_accuracy(cdist_softdtw(X_train[held], X_train[fit], gamma=g),
                                y_train[fit], y_train[held])
                for fit, held in splits]

    return cross_validate(args.gammas, score_one, "gamma", log=log)


def select_lr(X_train, y_train, exact: bool, args, log=print) -> tuple[float, dict]:
    splits = cv_splits(len(X_train), args.lr_splits, args.seed)

    def score_one(lr):
        out = []
        for fit, held in splits:
            D = fastcdtw_cdist(X_train[held], X_train[fit], exact=exact,
                               n_steps=args.nn_steps, lr=lr, n_mc=args.n_mc,
                               n_eval=args.n_eval,
                               chunk=args.chunk, device=args.device,
                               dtype=args.dtype, seed=args.seed)
            out.append(one_nn_accuracy(D, y_train[fit], y_train[held]))
        return out

    return cross_validate(args.lrs, score_one, "lr", log=log)


def distance_matrix(method: str, X_tr, y_tr, X_te, args, log=print):
    if method == "euclidean":
        return cdist_euclidean(X_te, X_tr), {}
    if method == "dtw":
        return cdist_dtw(X_te, X_tr, n_jobs=args.n_jobs), {}
    if method == "softdtw":
        if args.gamma is not None:
            gamma, scores = args.gamma, None
        else:
            gamma, scores = select_gamma(X_tr, y_tr, args, log=log)
        return (cdist_softdtw(X_te, X_tr, gamma=gamma),
                {"gamma": gamma, "gamma_scores": scores})
    if method in ("fastcdtw_exact", "fastcdtw_mc"):
        exact = method == "fastcdtw_exact"
        if args.lr is not None:
            lr, scores = args.lr, None
        else:
            lr, scores = select_lr(X_tr, y_tr, exact, args, log=log)
        D = fastcdtw_cdist(X_te, X_tr, exact=exact, n_steps=args.nn_steps, lr=lr,
                           n_mc=args.n_mc, n_eval=args.n_eval,
                           chunk=args.chunk,
                           device=args.device, dtype=args.dtype, seed=args.seed,
                           verbose=not args.quiet)
        extra = {"n_steps": args.nn_steps, "lr": lr, "lr_scores": scores,
                 "chunk": args.chunk}
        if not exact:
            extra |= {"n_mc": args.n_mc, "n_eval": args.n_eval}
        return D, extra
    raise ValueError(f"unknown method {method!r}")


def done_methods(task: str) -> dict:
    return read_locked(CHECKPOINT_PATH, {}).get(task, {}).get("methods", {})


def append_result(task: str, method: str, entry: dict, meta: dict):
    def mutate(ckpt):
        node = ckpt.setdefault(task, {})
        node.setdefault("_meta", {}).update(meta)
        node.setdefault("methods", {})[method] = entry
    update_json(CHECKPOINT_PATH, mutate)


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("datasets", nargs="*", help="explicit names; bypasses the filters")
    p.add_argument("--json", default=None, help="dataset list JSON ('datasets' key)")
    p.add_argument("--methods", nargs="+", default=METHODS, choices=METHODS)
    p.add_argument("--znorm", default="none", choices=["none", "series"])
    p.add_argument("--seed", type=int, default=42)

    c = p.add_argument_group("cost")
    c.add_argument("--max-pairs", type=int, default=2_000_000,
                   help="skip datasets whose n_train*n_test exceeds this; the "
                        "FastCDTW methods optimise one warping per pair")
    c.add_argument("--max-test", type=int, default=0,
                   help="subsample the test set to at most this many series "
                        "(0 = the full predefined split; anything else is a "
                        "documented deviation from the paper)")
    c.add_argument("--chunk", type=int, default=20000,
                   help="pairs optimised at once (0 = all)")
    c.add_argument("--n-jobs", type=int, default=-1, help="tslearn DTW workers")

    o = p.add_argument_group("ours")
    o.add_argument("--nn-steps", type=int, default=300,
                   help="Adam steps per pair; identical for every pair")
    o.add_argument("--lr", type=float, default=None,
                   help="fixed Adam lr; default is to cross-validate on train")
    o.add_argument("--lrs", type=float, nargs="+", default=[1e-3, 5e-3, 1e-2],
                   help="lr grid for the cross-validation")
    o.add_argument("--lr-splits", type=int, default=5,
                   help="random 2/3-1/3 splits per lr, matching the paper's "
                        "gamma protocol; lower it to cut cost, each split is a "
                        "full batch of warping fits")
    o.add_argument("--n-mc", type=int, default=800, help="MC draws per step")
    o.add_argument("--n-eval", type=int, default=20,
                   help="MC draws averaged into the reported distance")
    o.add_argument("--dtype", default="float32", choices=["float32", "float64"])

    s = p.add_argument_group("soft-DTW")
    s.add_argument("--gamma", type=float, default=None,
                   help="fixed gamma; default is to cross-validate on train")
    s.add_argument("--gammas", type=float, nargs="+", default=GAMMAS)
    s.add_argument("--gamma-splits", type=int, default=5,
                   help="random 2/3-1/3 splits per gamma (the paper's 5)")

    r = p.add_argument_group("run")
    r.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    r.add_argument("--list", action="store_true",
                   help="report sizes and pair counts, then exit")
    r.add_argument("--force", action="store_true",
                   help="recompute methods already in the checkpoint")
    r.add_argument("--quiet", action="store_true")

    return p.parse_args()


def select_datasets(names, args, log) -> tuple[list, list]:
    selected, rows = [], []
    explicit = bool(args.datasets)
    log(f"\n{'dataset':<28}{'n_tr':>7}{'n_te':>7}{'T':>7}{'K':>5}{'pairs':>12}   status")
    for name in names:
        try:
            X_tr, y_tr, X_te, y_te, classes = load_ucr_split(name, args.znorm)
        except Exception as e:
            log(f"{name:<28}{'-':>7}{'-':>7}{'-':>7}{'-':>5}{'-':>12}   LOAD FAILED: {e}")
            rows.append({"dataset": name, "status": f"load_failed: {e}"})
            continue
        if args.max_test and len(X_te) > args.max_test:
            rng = np.random.default_rng(args.seed)
            keep_idx = np.sort(rng.choice(len(X_te), args.max_test, replace=False))
            X_te, y_te = X_te[keep_idx], y_te[keep_idx]
        n_pairs = len(X_tr) * len(X_te)
        keep = explicit or n_pairs <= args.max_pairs
        status = "keep" if keep else f"skip (>{args.max_pairs:,} pairs)"
        if np.isnan(X_tr).any() or np.isnan(X_te).any():
            status, keep = "skip (NaN)", False
        log(f"{name:<28}{len(X_tr):>7}{len(X_te):>7}{X_tr.shape[1]:>7}"
            f"{len(classes):>5}{n_pairs:>12,}   {status}")
        rows.append({"dataset": name, "n_train": len(X_tr), "n_test": len(X_te),
                     "T": int(X_tr.shape[1]), "K": len(classes),
                     "pairs": int(n_pairs), "status": status})
        if keep:
            selected.append((name, X_tr, y_tr, X_te, y_te, classes))
    return selected, rows


def main():
    args = parse_args()
    if args.device == "cuda":
        import torch
        if not torch.cuda.is_available():
            raise SystemExit("CUDA not available; pass --device cpu.")
    for d in (CKPT_DIR, LOG_DIR):
        d.mkdir(parents=True, exist_ok=True)

    names = args.datasets or dataset_names(args.json)
    explicit = bool(args.datasets)
    lines = []

    def log(line=""):
        print(line, flush=True)
        lines.append(str(line))

    log(f"1-NN on UCR  |  methods={args.methods}  |  device={args.device}  "
        f"znorm={args.znorm}")
    lr_desc = (f"lr={args.lr:g}" if args.lr is not None
               else f"lr cross-validated over {args.lrs} x {args.lr_splits} splits")
    log(f"FastCDTW: softplus-cumsum warping, one knot per timestamp, "
        f"{args.nn_steps} Adam steps per pair, {lr_desc}")
    log(f"MC: n_mc={args.n_mc}, n_eval={args.n_eval}")
    log(f"Candidates: {len(names)}"
        + ("" if explicit else f"   filter: n_train*n_test <= {args.max_pairs:,}"))

    selected, rows = select_datasets(names, args, log)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    if not explicit:
        fields = ["dataset", "n_train", "n_test", "T", "K", "pairs", "status"]
        with open(SELECTION_PATH, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows([{k: r.get(k, "") for k in fields} for r in rows])
        log(f"\nDataset table -> {SELECTION_PATH}")

    log(f"Selected {len(selected)}/{len(names)}")
    if args.list:
        return
    if not selected:
        sys.exit("nothing to run -- raise --max-pairs or name datasets explicitly")

    t_start = time.perf_counter()
    for i, (name, X_tr, y_tr, X_te, y_te, classes) in enumerate(selected, 1):
        log(f"\n{'=' * 70}\n[{i}/{len(selected)}] {name}\n{'=' * 70}")
        majority = float(np.bincount(y_te).max() / len(y_te))
        log(f"  n_train={len(X_tr)}  n_test={len(X_te)}  T={X_tr.shape[1]}  "
            f"K={len(classes)}  chance={1 / len(classes):.3f}  majority={majority:.3f}")

        meta = {"n_train": len(X_tr), "n_test": len(X_te), "T": int(X_tr.shape[1]),
                "K": len(classes), "chance": round(1 / len(classes), 6),
                "majority": round(majority, 6), "znorm": args.znorm,
                "max_test": args.max_test}
        already = done_methods(name)

        for method in args.methods:
            if method in already and not args.force:
                log(f"  [{method}] already done: acc={already[method]['accuracy']:.4f}")
                continue
            t0 = time.perf_counter()
            D, extra = distance_matrix(method, X_tr, y_tr, X_te, args, log=log)
            acc = one_nn_accuracy(D, y_tr, y_te)
            elapsed = time.perf_counter() - t0
            append_result(name, method,
                          {"accuracy": acc, "time": float(elapsed),
                           "seed": args.seed, **extra}, meta)
            log(f"  [{method}] acc={acc:.4f}  time={elapsed:.1f}s")

    log(f"\n{'=' * 70}\nDone in {(time.perf_counter() - t_start) / 60:.1f} min.")
    log(f"Checkpoint -> {CHECKPOINT_PATH}")
    log(f"Build Table 2 with:  python3 {HERE / 'summarize.py'}")

    stem = (names[0] + "_ucr_nn") if (explicit and len(names) == 1) else "ucr_nn"
    (LOG_DIR / f"{stem}.txt").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
