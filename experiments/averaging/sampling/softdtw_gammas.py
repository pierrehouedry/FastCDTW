"""Soft-DTW drift for several gammas, next to DTW and FastCDTW from a sampling_linspace JSON.
Only Soft-DTW is recomputed (cheap); same series, same linspace grids, same normalisation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from sampling import PAIRS, drift, series, softdtw_value, trim

HERE = Path(__file__).resolve().parent


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--res", default=str(HERE / "results" / "sampling_linspace_seed0.json"))
    p.add_argument("--gammas", type=float, nargs="+", default=[1e-4, 1e-3, 1e-2, 1e-1, 1.0])
    p.add_argument("--min-n", type=int, default=100)
    p.add_argument("--max-n", type=int, default=5000)
    args = p.parse_args()

    f = Path(args.res)
    res = trim(json.loads(f.read_text()), args.min_n, args.max_n)
    ref_n = res["config"].get("ref_n", 200)
    t_ref = np.linspace(0.0, 1.0, ref_n)

    cols = ["DTW"] + [f"SDTW g={g:g}" for g in args.gammas] + ["FastCDTW"]
    lines = [f"Normalised drift ({f.stem}, n = {args.min_n}..{args.max_n}): mean ± std", "",
             f"{'pair':<13}" + "".join(f"{c:<20}" for c in cols)]
    for name, info in res["pairs"].items():
        fx, fy = (series(*s) for s in PAIRS[name])
        cells = [drift(info, "dtw")]
        for g in args.gammas:
            ref = softdtw_value(fx(t_ref), fy(t_ref), g)
            v = np.array([softdtw_value(fx(np.linspace(0, 1, r["n"])),
                                        fy(np.linspace(0, 1, r["n"])), g)
                          for r in info["rows"]])
            cells.append((v - ref) / info["ref"]["unaligned_dtw"])
        cells.append(drift(info, "fastcdtw"))
        lines.append(f"{name:<13}" + "".join(
            f"{f'{d.mean():.1e} ± {d.std():.1e}':<20}" for d in cells))
    txt = "\n".join(lines)
    out = f.with_name(f"{f.stem}_softdtw_gammas.txt")
    out.write_text(txt + "\n")
    print(txt)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
