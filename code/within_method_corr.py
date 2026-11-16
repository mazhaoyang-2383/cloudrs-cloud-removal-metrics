#!/usr/bin/env python
"""Recompute the pooled and within-method Spearman correlations of Table 3.

Table 3 of the manuscript reports four aggregation schemes for the same data.
Three of them are archived as CSV files in `tables/main/`:

    corr_method_level.csv    one coefficient per metric, computed over the 15 methods
    corr_sample_level.csv    one coefficient per metric, computed over the 6420 patches
    corr_highq_subset.csv    the same, over the nine-method qualified pool

The fourth, the within-method coefficient, is defined as the Spearman
correlation of the two sets of values after each has been centred on its own
method mean; it removes the between-method component that dominates the pooled
coefficient. It is computed here from the two per-patch tables and written back
to `tables/main/corr_within_method.csv`, so that the column in the manuscript
can be traced to a file in this archive.

Inputs:
    tables/main/per_patch_metrics.csv     PSNR, SSIM, SAM, ERGAS, MAE, RMSE, method, patch
    tables/main/per_patch_downstream.csv  dLAI_mean, method, patch

Outputs:
    tables/main/corr_within_method.csv    pooled, within-method and mean per-method
                                          coefficients for the six metrics
    stdout: the per-method coefficients quoted in Sections 3.2 and 4.1

Pure standard library on purpose: no numpy, pandas or scipy required, and the
tie handling of `rankdata(method="average")` is reimplemented below so that the
result does not depend on a scipy version.

Usage (from the package root):
    python code/within_method_corr.py
    python code/within_method_corr.py --tables tables/main --csv tables/main/corr_within_method.csv
    python code/within_method_corr.py --no-csv          # print only
"""

import argparse
import csv
import math
import os
from collections import defaultdict

METRICS = ("PSNR", "SSIM", "SAM", "ERGAS", "MAE", "RMSE")


def read_inputs(tables_dir):
    metrics, downstream = {}, {}
    with open(os.path.join(tables_dir, "per_patch_metrics.csv")) as fh:
        for row in csv.DictReader(fh):
            metrics[(row["method"], row["patch"])] = {m: float(row[m]) for m in METRICS}
    with open(os.path.join(tables_dir, "per_patch_downstream.csv")) as fh:
        for row in csv.DictReader(fh):
            downstream[(row["method"], row["patch"])] = float(row["dLAI_mean"])
    keys = [k for k in metrics if k in downstream]
    return metrics, downstream, keys


def rank(values):
    """Average ranks, ties handled as in scipy.stats.rankdata(method='average')."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    out = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            out[order[k]] = avg
        i = j + 1
    return out


def pearson(a, b):
    n = len(a)
    ma, mb = sum(a) / n, sum(b) / n
    sa = math.sqrt(sum((v - ma) ** 2 for v in a))
    sb = math.sqrt(sum((v - mb) ** 2 for v in b))
    if sa == 0 or sb == 0:
        return float("nan")          # undefined, e.g. SAM of zero filling
    return sum((a[i] - ma) * (b[i] - mb) for i in range(n)) / (sa * sb)


def spearman(a, b):
    return pearson(rank(a), rank(b))


def coefficients(metrics, downstream, keys):
    """Pooled, within-method and mean per-method coefficient for each metric."""
    methods = sorted({k[0] for k in keys})
    y_all = [downstream[k] for k in keys]
    rows = []
    for m in METRICS:
        x = [metrics[k][m] for k in keys]
        pooled = spearman(x, y_all)

        per_x, per_y = defaultdict(list), defaultdict(list)
        for k, v in zip(keys, x):
            per_x[k[0]].append(v)
            per_y[k[0]].append(downstream[k])
        mx = {g: sum(v) / len(v) for g, v in per_x.items()}
        my = {g: sum(v) / len(v) for g, v in per_y.items()}
        xc = [v - mx[k[0]] for k, v in zip(keys, x)]
        yc = [downstream[k] - my[k[0]] for k in keys]
        within = spearman(xc, yc)

        per = [spearman(per_x[g], per_y[g]) for g in methods]
        per = [p for p in per if not math.isnan(p)]
        rows.append({
            "metric": m,
            "pooled_patch_level": pooled,
            "within_method": within,
            "mean_per_method": sum(per) / len(per),
            "n_methods_with_defined_rho": len(per),
        })
    return rows, methods


def write_csv(path, rows):
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["metric", "pooled_patch_level", "within_method",
                    "mean_per_method", "n_methods_with_defined_rho"])
        for r in rows:
            w.writerow([r["metric"]] +
                       ["%.3f" % r[k] if not math.isnan(r[k]) else "nan"
                        for k in ("pooled_patch_level", "within_method", "mean_per_method")] +
                       [r["n_methods_with_defined_rho"]])


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tables", default=os.path.join("tables", "main"),
                    help="directory holding the two per-patch tables")
    ap.add_argument("--csv", default=None,
                    help="write the coefficients here (default: <tables>/corr_within_method.csv)")
    ap.add_argument("--no-csv", action="store_true", help="print only, write nothing")
    args = ap.parse_args()

    metrics, downstream, keys = read_inputs(args.tables)
    rows, methods = coefficients(metrics, downstream, keys)

    print("method-patch pairs: %d   methods: %d" % (len(keys), len(methods)))
    print("%-6s %8s %8s %16s" % ("metric", "pooled", "within", "mean per method"))
    for r in rows:
        print("%-6s %+8.3f %+8.3f %+16.3f" % (r["metric"], r["pooled_patch_level"],
                                              r["within_method"], r["mean_per_method"]))

    print("\nper method (SAM, SSIM, ERGAS, PSNR):")
    for g in methods:
        ks = [k for k in keys if k[0] == g]
        row = {m: spearman([metrics[k][m] for k in ks], [downstream[k] for k in ks])
               for m in ("SAM", "SSIM", "ERGAS", "PSNR")}
        print("  %-15s " % g + "  ".join(
            "%s=%+.3f" % (m, row[m]) if not math.isnan(row[m]) else "%s=  n/a" % m
            for m in row))

    if not args.no_csv:
        out = args.csv or os.path.join(args.tables, "corr_within_method.csv")
        write_csv(out, rows)
        print("\nwritten: %s" % out)


if __name__ == "__main__":
    main()
