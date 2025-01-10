
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plot_likert
from scipy.stats import ttest_ind, linregress, mannwhitneyu, shapiro, wilcoxon


def print_results(x1, x2, precision=1, f=ttest_ind):
    ttest = f(x1, x2)
    print(
        f"{x1.mean():.{precision}f}, "
        f"{x1.std():.{precision}f} SD"
        f" vs "
        f"{x2.mean():.{precision}f}, "
        f"{x2.std():.{precision}f} SD, "
        f"p={ttest.pvalue:.8f}"
    )


def main():
    results_dir = Path("results/")
    out_dir = results_dir / "likert"
    out_dir.mkdir(exist_ok=True)
    results_dirs = results_dir.glob("AGE*")

    meta_df = results_dir / "results.csv"
    meta_df = pd.read_csv(meta_df)
    meta_df.set_index("study_id", inplace=True)

    cols = ["confidence_0", "confidence_1", "confidence_2"]
    dfs = []
    for indiv_dir in results_dirs:
        study_id = indiv_dir.stem
        explain_helpful = meta_df.loc[study_id, "explain_helpful"]
        explain_helpful = explain_helpful >= 4
        df = pd.read_csv(indiv_dir / "study_results.csv")
        df = df.loc[:, cols]
        df["explain_helpful"] = explain_helpful
        dfs.append(df)

    df = pd.concat(dfs)
    df = df.fillna(2)

    df.columns = ["Stage 1", "Stage 2", "Stage 3", "explain_helpful"]

    df.loc[:, ["Stage 1", "Stage 2", "Stage 3"]] = df.loc[:, ["Stage 1", "Stage 2", "Stage 3"]] + 1
    helpful = df.loc[df["explain_helpful"]]
    unhelpful = df.loc[~df["explain_helpful"]]

    for i, test in enumerate([ttest_ind, mannwhitneyu, wilcoxon]):
        print("All confidence")
        print_results(df["Stage 1"], df["Stage 2"], 2, test)
        print_results(df["Stage 2"], df["Stage 3"], 2, test)
        print("Helpful")
        print_results(helpful["Stage 2"], helpful["Stage 3"], 2, test)
        print("Unhelpful")
        print_results(unhelpful["Stage 2"], unhelpful["Stage 3"], 2, test)
        if i != 2:
            print("Helpful vs Unhelpful")
            print_results(helpful["Stage 2"], unhelpful["Stage 2"], 2, test)
            print_results(helpful["Stage 3"], unhelpful["Stage 3"], 2, test)

    test1 = mannwhitneyu(helpful["Stage 2"], unhelpful["Stage 2"])
    test2 = mannwhitneyu(helpful["Stage 3"], unhelpful["Stage 3"])

    scale = [
        "1 - Not confident at all",
        "2",
        "3",
        "4",
        "5 - Very confident"
    ]
    num_scale = [1, 2, 3, 4, 5]
    for i in num_scale:
        df = df.applymap(lambda x: scale[i] if i == x else x)

    ax = plot_likert.plot_likert(df, scale, figsize=(9, 2))
    plt.tight_layout()
    plt.savefig(out_dir / "confidence_likert.pdf")

    print("Done!")


if __name__ == "__main__":
    main()
