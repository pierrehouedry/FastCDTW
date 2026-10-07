from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
CHECKPOINT_PATH = HERE / "results" / "checkpoints" / "ucr_nn_checkpoint.json"
ORDER = ["euclidean", "dtw", "softdtw", "fastcdtw_exact", "fastcdtw_mc"]
LABEL = {"euclidean": "Euc.", "dtw": "DTW", "softdtw": "SDTW",
         "fastcdtw_exact": "FC-ex", "fastcdtw_mc": "FC-mc"}


def load(path: Path) -> dict[str, dict[str, float]]:
    """{dataset: {method: accuracy}}."""
    if not path.exists():
        raise SystemExit(f"no checkpoint at {path} -- run run.py first")
    out = {}
    for name, node in json.loads(path.read_text()).items():
        accs = {m: e["accuracy"] for m, e in node.get("methods", {}).items()
                if "accuracy" in e}
        if accs:
            out[name] = accs
    return out


def common(acc: dict[str, dict[str, float]], methods: list[str]) -> list[str]:
    return [d for d, a in acc.items() if all(m in a for m in methods)]


def win_matrix(acc, datasets, methods, tolerance: float) -> np.ndarray:
    m = len(methods)
    out = np.full((m, m), np.nan)
    for i, a in enumerate(methods):
        for j, b in enumerate(methods):
            if i != j:
                out[i, j] = 100.0 * float(np.mean(
                    [acc[d][a] >= tolerance * acc[d][b] for d in datasets]))
    return out


def render(matrix: np.ndarray, methods: list[str], n: int, tolerance: float) -> str:
    head = f"{'A (v) vs B (>)':<16}" + "".join(f"{LABEL.get(m, m):>10}" for m in methods)
    lines = [f"1-NN, {n} datasets, within {tolerance:.0%} or better", "", head,
             "-" * len(head)]
    for i, a in enumerate(methods):
        cells = "".join("         -" if np.isnan(matrix[i, j])
                        else f"{matrix[i, j]:>10.2f}" for j in range(len(methods)))
        lines.append(f"{LABEL.get(a, a):<16}" + cells)
    return "\n".join(lines)


def render_accuracies(acc, datasets, methods, tolerance: float) -> str:
    """Per-dataset table; '*' marks every method within `tolerance` of the row's
    best, compared on the printed 3-decimal values so the marks can be checked
    from the table itself (this is the bold rule of the paper's appendix tables)."""
    head = f"{'dataset':<28}" + "".join(f"{LABEL.get(m, m):>10}" for m in methods)
    lines = [head, "-" * len(head)]
    for d in sorted(datasets):
        shown = {m: round(acc[d][m], 3) for m in methods}
        best = max(shown.values())
        cells = "".join(f"{acc[d][m]:>9.3f}"
                        + ("*" if shown[m] >= tolerance * best - 1e-9 else " ")
                        for m in methods)
        lines.append(f"{d:<28}" + cells)
    means = [np.mean([acc[d][m] for d in datasets]) for m in methods]
    lines += ["-" * len(head),
              f"{'mean':<28}" + "".join(f"{v:>9.3f} " for v in means)]
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", default=str(CHECKPOINT_PATH))
    p.add_argument("--methods", nargs="+", default=None,
                   help="subset to compare; default is every method present")
    p.add_argument("--tolerance", type=float, default=0.99,
                   help="A counts as beating B when acc_A >= tolerance * acc_B")
    p.add_argument("--csv", action="store_true", help="also write CSV files")
    args = p.parse_args()

    acc = load(Path(args.checkpoint))
    present = [m for m in ORDER if any(m in a for a in acc.values())]
    present += sorted({m for a in acc.values() for m in a} - set(present))
    methods = args.methods or present

    datasets = common(acc, methods)
    if not datasets:
        raise SystemExit(f"no dataset has results for all of {methods}; "
                         f"present: {present}")
    incomplete = len(acc) - len(datasets)

    matrix = win_matrix(acc, datasets, methods, args.tolerance)
    table = render(matrix, methods, len(datasets), args.tolerance)
    per_ds = render_accuracies(acc, datasets, methods, args.tolerance)

    print(table)
    if incomplete:
        print(f"\n({incomplete} dataset(s) skipped -- not every method has a "
              f"result yet)")
    print("\n" + per_ds)

    out_dir = Path(args.checkpoint).parent.parent
    (out_dir / "table2_ucr_nn.txt").write_text(table + "\n\n" + per_ds + "\n")
    print(f"\nTable -> {out_dir / 'table2_ucr_nn.txt'}")

    if args.csv:
        with open(out_dir / "table2_ucr_nn.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["A_vs_B"] + methods)
            for i, a in enumerate(methods):
                w.writerow([a] + ["" if np.isnan(v) else f"{v:.2f}" for v in matrix[i]])
        with open(out_dir / "accuracy_ucr_nn.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["dataset"] + methods)
            for d in sorted(datasets):
                w.writerow([d] + [f"{acc[d][m]:.4f}" for m in methods])
        print(f"CSVs  -> {out_dir}")


if __name__ == "__main__":
    main()
