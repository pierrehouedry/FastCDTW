"""
Wall-clock of ONE cross-distance matrix per (dataset, method).
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics as st
import time
from pathlib import Path

import numpy as np

from checkpoint import read_locked, update_json
from data import load_ucr_split
from distances import (cdist_dtw, cdist_euclidean, cdist_softdtw,
                       one_nn_accuracy)

HERE = Path(__file__).resolve().parent
CKPT_DIR = HERE / "results" / "checkpoints"
SOURCE_PATH = CKPT_DIR / "ucr_nn_checkpoint.json"
TIMING_PATH = CKPT_DIR / "ucr_nn_timing.json"
TIMING_PATH_CUDA = CKPT_DIR / "ucr_nn_timing_cuda.json"


def timing_path(args) -> Path:
    if args.out:
        q = Path(args.out)
        return q if q.is_absolute() else HERE / q
    return TIMING_PATH_CUDA if args.device == "cuda" else TIMING_PATH

METHODS = ["euclidean", "dtw", "softdtw", "fastcdtw_exact", "fastcdtw_mc"]


def selected_hyperparameters(source: dict, name: str) -> dict:
    methods = source.get(name, {}).get("methods", {})
    out = {}
    if "softdtw" in methods and methods["softdtw"].get("gamma") is not None:
        out["gamma"] = float(methods["softdtw"]["gamma"])
    for m in ("fastcdtw_exact", "fastcdtw_mc"):
        if m in methods and methods[m].get("lr") is not None:
            out[m] = float(methods[m]["lr"])
    return out


def timed_matrix(method: str, X_tr, y_tr, X_te, hp: dict, args):
    import torch

    cuda = args.device == "cuda"

    def run():
        if method == "euclidean":
            return cdist_euclidean(X_te, X_tr)
        if method == "dtw":
            return cdist_dtw(X_te, X_tr, n_jobs=args.n_jobs)
        if method == "softdtw":
            return cdist_softdtw(X_te, X_tr, gamma=hp["gamma"])
        from distances import fastcdtw_cdist
        return fastcdtw_cdist(X_te, X_tr, exact=(method == "fastcdtw_exact"),
                              n_steps=args.nn_steps, lr=hp[method],
                              n_mc=args.n_mc, n_eval=args.n_eval,
                              chunk=args.chunk, device=args.device,
                              dtype=args.dtype, seed=args.seed, verbose=False)

    if cuda:
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    D = run()
    if cuda:
        torch.cuda.synchronize()
    return D, time.perf_counter() - t0


def environment(args) -> dict:
    import torch
    cpu_model = ""
    try:
        for line in open("/proc/cpuinfo"):
            if line.startswith("model name"):
                cpu_model = line.split(":", 1)[1].strip()
                break
    except OSError:
        pass
    env = {"device": args.device, "n_jobs": args.n_jobs, "cpu": cpu_model,
           "torch_threads": torch.get_num_threads(),
           "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
           "cpu_count": os.cpu_count(), "host": platform.node()}
    if args.device == "cuda" and torch.cuda.is_available():
        env["gpu"] = torch.cuda.get_device_name(0)
    return env


def summarize(path: Path) -> None:
    timing = read_locked(path, {})
    if not timing:
        raise SystemExit(f"nothing in {path}")
    print(f"# {path}")
    by_env: dict[str, dict[str, list]] = {}
    for name, node in timing.items():
        for method, entry in node.get("methods", {}).items():
            env = entry.get("env", {})
            key = (f"device={env.get('device')} n_jobs={env.get('n_jobs')} "
                   f"threads={env.get('torch_threads')} cpu={env.get('cpu')}"
                   + (f" gpu={env['gpu']}" if env.get("gpu") else ""))
            by_env.setdefault(key, {}).setdefault(method, []).append(entry["time"])
    for key, per_method in by_env.items():
        print(f"\n{key}")
        print(f"{'method':<18}{'n':>5}{'mean':>12}{'median':>12}{'max':>12}")
        for method in METHODS:
            ts = per_method.get(method)
            if not ts:
                continue
            print(f"{method:<18}{len(ts):>5}{st.mean(ts):>12.2f}"
                  f"{st.median(ts):>12.2f}{max(ts):>12.2f}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("datasets", nargs="*",
                   help="names to time; default is every dataset in the source checkpoint")
    p.add_argument("--methods", nargs="+", default=METHODS, choices=METHODS)
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    p.add_argument("--n-jobs", type=int, default=1,
                   help="tslearn DTW workers; 1 matches cdist_soft_dtw, which is "
                        "single-threaded and takes no n_jobs")
    p.add_argument("--torch-threads", type=int, default=None,
                   help="torch.set_num_threads for the FastCDTW rows on CPU")
    p.add_argument("--repeats", type=int, default=1,
                   help="time the matrix this many times; the minimum is kept")
    p.add_argument("--znorm", default="none", choices=["none", "series"])
    p.add_argument("--nn-steps", type=int, default=300)
    p.add_argument("--n-mc", type=int, default=800)
    p.add_argument("--n-eval", type=int, default=20)
    p.add_argument("--chunk", type=int, default=20000)
    p.add_argument("--dtype", default="float32", choices=["float32", "float64"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default=None,
                   help="timing checkpoint to read/write; default is "
                        "results/checkpoints/ucr_nn_timing.json on cpu and "
                        "ucr_nn_timing_cuda.json on cuda")
    p.add_argument("--force", action="store_true")
    p.add_argument("--summarize", action="store_true")
    args = p.parse_args()
    out_path = timing_path(args)

    if args.summarize:
        summarize(out_path)
        return

    import torch
    if args.torch_threads:
        torch.set_num_threads(args.torch_threads)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA not available; pass --device cpu.")

    source = read_locked(SOURCE_PATH, {})
    if not source:
        raise SystemExit(f"no source checkpoint at {SOURCE_PATH}")
    names = args.datasets or sorted(source)
    env = environment(args)
    print(f"timing final matrices only | {env}\ncheckpoint: {out_path}")

    for name in names:
        hp = selected_hyperparameters(source, name)
        done = read_locked(out_path, {}).get(name, {}).get("methods", {})
        todo = [m for m in args.methods if args.force or m not in done]
        if not todo:
            print(f"{name}: already done")
            continue
        try:
            X_tr, y_tr, X_te, y_te, classes = load_ucr_split(name, args.znorm)
        except Exception as e:
            print(f"{name}: LOAD FAILED: {e}")
            continue
        print(f"{name}: n_train={len(X_tr)} n_test={len(X_te)} T={X_tr.shape[1]}")

        for method in todo:
            if method == "softdtw" and "gamma" not in hp:
                print(f"  [{method}] no selected gamma in the source checkpoint; skipped")
                continue
            if method.startswith("fastcdtw") and method not in hp:
                print(f"  [{method}] no selected lr in the source checkpoint; skipped")
                continue
            times = []
            for _ in range(args.repeats):
                D, dt = timed_matrix(method, X_tr, y_tr, X_te, hp, args)
                times.append(dt)
            acc = one_nn_accuracy(D, y_tr, y_te)
            stored = source[name]["methods"].get(method, {}).get("accuracy")
            entry = {"time": min(times), "times": times, "accuracy": acc,
                     "accuracy_in_source": stored, "env": env,
                     "hyperparameters": {k: v for k, v in hp.items()
                                         if k == "gamma" or k == method},
                     "n_steps": args.nn_steps, "n_mc": args.n_mc,
                     "n_eval": args.n_eval, "seed": args.seed}

            def mutate(ckpt, name=name, method=method, entry=entry):
                node = ckpt.setdefault(name, {})
                node.setdefault("_meta", {}).update(
                    {"n_train": len(X_tr), "n_test": len(X_te),
                     "T": int(X_tr.shape[1]), "K": len(classes)})
                node.setdefault("methods", {})[method] = entry

            update_json(out_path, mutate)
            flag = "" if stored is None or abs(acc - stored) < 1e-9 else \
                   f"  (accuracy differs from source: {stored:.4f})"
            print(f"  [{method}] {min(times):8.2f}s  acc={acc:.4f}{flag}")

    print(f"\n-> {out_path}\nSummary:  python3 {Path(__file__).name} --summarize --device {args.device}")


if __name__ == "__main__":
    main()
