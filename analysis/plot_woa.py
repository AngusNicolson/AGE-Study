
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy.stats import ttest_ind


def main():
    results_dir = Path("results/")
    out_dir = results_dir / "trust"
    out_dir.mkdir(exist_ok=True)
    results_dirs = list(results_dir.glob("AGE*"))
    file_type = ".png"

    cols = []
    for suffix in ["", "_no_thresh"]:
        for k in ["woa_12", "woa_13", "woa_23"]:
            cols.append(k + suffix)

    columns = ["Stage 2 WoA", "Stage 3 WoA", "Partial WoA"]
    columns = columns + [v + suffix for v in columns]
    dfs = []
    for indiv_dir in results_dirs:
        df = pd.read_csv(indiv_dir / "study_results.csv")
        df = df.loc[:, cols]
        df.columns = columns
        dfs.append(df)

    df = pd.concat(dfs)

    for col in columns:
        print(col, df[col].mean())

    for col in columns:
        sns.displot(df[col])
        plt.show()

    cols = ["Stage 2 WoA", "Stage 3 WoA", "Partial WoA"]
    fig, ax = plt.subplots(figsize=(7, 3))
    vals = df[cols].values
    vals = [v[~np.isnan(v)] for v in vals.T]
    ax.boxplot(vals, vert=False, widths=0.5)
    ax.set_yticklabels(cols)
    ax.set_xlabel("Weight of Advice")
    plt.tight_layout()
    plt.savefig(out_dir / f"woa_boxplot{file_type}", bbox_inches="tight")

    xlim = [-1, 2]
    delta = 0.25
    bins = np.arange(xlim[0], xlim[1] + delta + 0.05, delta) - delta/2
    xlim[0] = xlim[0] - delta
    xlim[1] = xlim[1] + delta
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(9, 3))
    sns.histplot(data=df["Stage 2 WoA"], ax=axes[0], bins=bins)
    axes[0].set_xlim(xlim)
    sns.histplot(data=df["Stage 3 WoA"], ax=axes[1], bins=bins)
    axes[1].set_xlim(xlim)
    sns.histplot(data=df["Partial WoA"], ax=axes[2], bins=bins)
    axes[2].set_xlim(xlim)
    plt.tight_layout()
    plt.savefig(out_dir / f"woa_dist_zoom{file_type}", bbox_inches="tight")
    plt.show()

    xlim = [-4, 5]
    delta = 0.5
    bins = np.arange(xlim[0], xlim[1] + delta + 0.05, delta) - delta/2
    xlim[0] = xlim[0] - delta
    xlim[1] = xlim[1] + delta
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(9, 3))
    sns.histplot(data=df["Stage 2 WoA"], ax=axes[0], bins=bins)
    axes[0].set_xlim(xlim)
    sns.histplot(data=df["Stage 3 WoA"], ax=axes[1], bins=bins)
    axes[1].set_xlim(xlim)
    sns.histplot(data=df["Partial WoA"], ax=axes[2], bins=bins)
    axes[2].set_xlim(xlim)
    plt.tight_layout()
    plt.savefig(out_dir / f"woa_dist{file_type}", bbox_inches="tight")
    plt.show()

    cols = ["Stage 3 WoA", "Stage 2 WoA"] # ["Stage 2 WoA", "Stage 3 WoA"]  # , "Partial WoA"]
    labels = ["Stage 3", "Stage 2"]
    mean_woas = [v[cols].mean() for v in dfs]
    mean_woas = pd.concat(mean_woas, axis=1).T

    fig, ax = plt.subplots(figsize=(7, 2))
    ax.boxplot(mean_woas, vert=False, widths=0.5)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Weight of Advice")
    plt.tight_layout()
    plt.savefig(out_dir / f"woa_participant_boxplot{file_type}", bbox_inches="tight")

    def print_ttest_results(x1, x2):
        ttest = ttest_ind(x1, x2)
        print(
            f"{x1.mean():.3f}, "
            f"{x1.std():.3f} SD"
            f" vs "
            f"{x2.mean():.3f}, "
            f"{x2.std():.3f} SD, "
            f"p={ttest.pvalue:.8f}"
        )

    # not significant
    ttest_participant = ttest_ind(mean_woas["Stage 2 WoA"], mean_woas["Stage 3 WoA"])
    print_ttest_results(mean_woas["Stage 2 WoA"], mean_woas["Stage 3 WoA"])

    # not significant
    cols = ["Stage 3 WoA", "Stage 2 WoA"]
    vals = df[cols].values
    vals = [v[~np.isnan(v)] for v in vals.T]
    ttest = ttest_ind(vals[1], vals[0])
    print_ttest_results(vals[1], vals[0])

    print("Done!")


if __name__ == "__main__":
    main()
