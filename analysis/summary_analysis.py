
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import ttest_ind, linregress, mannwhitneyu, ttest_rel, wilcoxon
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
from analyse_results import bootstrap_mae

low_bins = np.array([13] + list(np.arange(16, 39, 2)))
high_bins = np.array(list(np.arange(16, 39, 2)) + [42])


def main():
    base_dir = Path("/home/lina3782/labs/protopnet/AGE/results/")
    out_dir = base_dir / "summary"
    out_dir.mkdir(exist_ok=True)
    csv_paths = [v / "study_results.csv" for v in base_dir.glob("AGE*")]

    jisc_dir = Path("/home/lina3782/labs/protopnet/AGE/study_data/jisc")
    jisc_paths = list(jisc_dir.glob("*.csv"))
    jisc_paths.sort()

    dfs = [load_csv(v, v.parent.name) for v in csv_paths]
    df = pd.concat(dfs)

    jisc_dfs = [pd.read_csv(path) for path in jisc_paths]
    jisc_dfs[0] = process_jisc_stage_1(jisc_dfs[0])
    jisc_dfs[1] = process_jisc_stage_2(jisc_dfs[1])
    jisc_dfs[2] = process_jisc_stage_3(jisc_dfs[2])

    n_participants = len(dfs)

    ages = dfs[0]["age"].values
    bins = [13] + list(np.arange(16, 39, 2)) + [42]
    sns.displot(ages, bins=bins)
    plt.ylabel("No. images")
    plt.xlabel("Gestation Age / weeks")
    plt.savefig(out_dir / "AGE_study_age_dist.pdf", bbox_inches='tight')

    study_ids = []
    maes = []
    for i in range(3):
        mae = df.groupby("study_id")[f"exp_error_{i}"].agg(lambda x: abs(x).mean())
        maes.append(mae.values)
        study_ids.append(mae.index.values)
    maes = np.array(maes)
    maes = maes*7

    if not (study_ids[0] == study_ids).all():
        raise ValueError("Study IDs not the same for different stages!")

    study_ids = study_ids[0]
    #plt.xticks(rotation=45, ha='right')

    delta_1 = maes[0, :] - maes[1, :]
    delta_2 = maes[1, :] - maes[2, :]
    print(delta_1)
    print(delta_2)
    print(
        (delta_1 < delta_2).sum(),
        "participants had a greater improvement in stage 3 than stage 2."
    )

    df["pred_agrees_0"] = (
            (df["exp_0"] >= low_bins[df["prediction"]]) &
            (df["exp_0"] <= high_bins[df["prediction"]])
    )
    df["pred_agrees_1"] = (
            (df["exp_1"] >= low_bins[df["prediction"]]) &
            (df["exp_1"] <= high_bins[df["prediction"]])
    )
    df["pred_agrees_2"] = (
            (df["exp_2"] >= low_bins[df["prediction"]]) &
            (df["exp_2"] <= high_bins[df["prediction"]])
    )

    plt.rcParams.update({'font.size': 14})

    fig, ax = plt.subplots()
    ax.scatter(df["age"], df["exp_0"], alpha=0.7, marker=".")
    plt.show()

    stage_mapping = {'exp_0': 'Stage 1', 'exp_1': 'Stage 2', 'exp_2': 'Stage 3'}
    melted_df = df.loc[:, ["age", "exp_0", "exp_1", "exp_2"]].melt(id_vars='age', var_name='variable', value_name='value')
    melted_df['Stage'] = melted_df['variable'].map(stage_mapping)
    melted_df.columns = ["Ground Truth GA / days", "exp", "Predicted GA / days", "Stage"]

    sns.displot(melted_df, x="Ground Truth GA / days", y="Predicted GA / days", col="Stage")
    plt.show()

    sns.displot(melted_df, x="Ground Truth GA / days", y="Predicted GA / days", col="Stage", kind="kde")
    plt.show()

    sns.jointplot(data=melted_df, x="Ground Truth GA / days", y="Predicted GA / days", hue="Stage")
    plt.show()

    sns.jointplot(data=df.reset_index(), x="age", y="exp_0")
    plt.show()

    sns.displot(df.reset_index(), x="age", y="exp_0", kind="kde")
    plt.show()

    handles = [
        mlines.Line2D([], [], color=f'C0', marker='o', linewidth=0, markersize=6,
                      label=f'Clinician'),
        mlines.Line2D([], [], color=f'C1', marker='o', linewidth=0, markersize=6, label=f'Model'),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    for i, ax in enumerate(axes):
        ax.scatter(df["age"], df[f"exp_{i}"], alpha=0.5)
        ax.scatter(df["age"], df["age_pred"], alpha=0.5)
        #ax.axis("equal")
        ax.set_ylim([12, 42])
        ax.set_xlim([14, 42])
        ax.set_aspect("equal")
        ax.set_xlabel("Ground Truth GA / weeks")
    axes[0].set_ylabel("Predicted GA / weeks")
    plt.legend(handles=handles, frameon=False)
    plt.tight_layout()
    plt.savefig(out_dir / "association_plot.pdf", bbox_inches="tight")
    plt.show()

    fig, ax = plt.subplots()
    for i in range(3):
        ax.scatter(range(n_participants), maes[i], label=f"Stage {i+1}", marker="x")
    plt.legend()
    plt.xlabel("Participant")
    plt.ylabel("MAE / days")
    plt.savefig(out_dir / "mae_scatter_plot.png", bbox_inches="tight")

    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    for i in range(3):
        ax.scatter(study_ids, maes[i], label=f"Stage {i + 1}", marker="x")
    plt.xticks(rotation=45, ha="right")
    plt.legend()
    plt.xlabel("Participant")
    plt.ylabel("MAE / days")
    plt.tight_layout()
    plt.savefig(out_dir / "mae_scatter_plot_names.png", bbox_inches="tight")

    ttest0 = ttest_ind(maes[0], maes[1])
    ttest1 = ttest_ind(maes[1], maes[2])

    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    bar = plt.bar(["1", "2", "3"], maes.mean(axis=1), yerr=maes.std(axis=1))
    ax.bar_label(bar, fmt="%.1f")
    #plt.ylabel("Mean Absolute Error / days")
    plt.ylabel("MAE / days")
    #plt.xlabel("Stage")
    ax.set_xticklabels(["Stage 1", "Stage 2", "Stage 3"])
    ax.set_ylim([0, 30])
    plt.text(0.40, 18, f"p={ttest0.pvalue:.3f}", color="red", size=10)
    plt.text(1.35, 17, f"p={ttest1.pvalue:.3f}", color="red", size=10)
    plt.tight_layout()
    plt.savefig(out_dir / "mae_bars.pdf", bbox_inches="tight")
    plt.show()

    def plot_mae_lines(striped_idx=(0, 1, 3), labels=None, legend=True, figsize=(10, 5.2)):
        linestyles = ["-" for i in range(n_participants)]
        for i in striped_idx:
            linestyles[i] = "--"

        fig, ax = plt.subplots(figsize=figsize)
        for i in range(n_participants):
            if labels is None:
                label = study_ids[i]
            else:
                label = labels[i]
            ax.plot(
                [j + 1 for j in range(3)],
                maes[:, i],
                label=label,
                marker="x",
                linestyle=linestyles[i]
            )
        #plt.xlabel("Stage")
        ax.set_xticks([j + 1 for j in range(3)])
        ax.set_xticklabels(["Stage 1", "Stage 2", "Stage 3"])
        plt.ylabel("MAE / days")
        if legend:
            fig.legend(loc=7)
            fig.subplots_adjust(right=0.75)
        return fig, ax

    stage_3_worse_idx = [0, 1, 3, 5]
    fig, ax = plot_mae_lines(stage_3_worse_idx)
    plt.savefig(out_dir / "mae_line_plot.png", bbox_inches="tight")

    mae_df = pd.DataFrame(maes.T, columns=["Stage 1", "Stage 2", "Stage 3"])
    mae_df.index = study_ids
    mae_df = mae_df.reset_index()
    mae_df = mae_df.rename(columns={"index": "study_id"})

    for i in range(3):
        mae_df = pd.merge(mae_df, jisc_dfs[i], on="study_id", suffixes=("", f"_{i}"))

    mae_df["years_ultrasound_text"] = mae_df.loc[:, "years_ultrasound"].copy()
    years_text_2_num = {
        "< 2 years": 0,
        "2-5 years": 1,
        "6-10 years": 2,
        "≥10 years": 3,
    }
    mae_df["years_ultrasound"] = mae_df["years_ultrasound"].replace(years_text_2_num)

    mae_df["mae_delta"] = mae_df["Stage 3"] - mae_df["Stage 2"]
    free_text_df = mae_df.sort_values("mae_delta").loc[:,
                   ["study_id", "mae_delta", "explanations_free_text",
                    "outside_head_free_text"]]
    free_text_df.reset_index(inplace=True, drop=True)
    free_text_df.to_csv(out_dir / "free_text_summary.csv")
    study_id_order = free_text_df.study_id.values

    trust_combo_cols = ["confident", "predictable", "reliable", "safe_to_rely", "like"]
    trust_combo_neg_cols = ["wary"]
    mae_df["trust_combo"] = 0
    for col in trust_combo_cols:
        mae_df["trust_combo"] += mae_df[col]
    for col in trust_combo_neg_cols:
        mae_df["trust_combo"] += -mae_df[col]
    mae_df["trust_combo_2"] = 0
    for col in trust_combo_cols:
        mae_df["trust_combo_2"] += mae_df[col + "_2"]
    for col in trust_combo_neg_cols:
        mae_df["trust_combo_2"] += -mae_df[col + "_2"]
    max_trust_combo = len(trust_combo_cols)*5 - len(trust_combo_neg_cols)*1
    min_trust_combo = len(trust_combo_cols)*1 - len(trust_combo_neg_cols)*5
    mae_df["trust_combo"] = mae_df["trust_combo"] / max_trust_combo
    mae_df["trust_combo_2"] = mae_df["trust_combo_2"] / max_trust_combo

    stage_3_worse_maes = mae_df.iloc[stage_3_worse_idx]
    stage_3_better_maes = mae_df.loc[~mae_df["study_id"].isin(stage_3_worse_maes.study_id)]

    plt.rcParams["font.size"] = 16
    explain_unhelpful_idx = mae_df.loc[mae_df["explain_helpful"] <= 3].index.values
    fig, ax = plot_mae_lines(explain_unhelpful_idx, legend=False)
    plt.savefig(out_dir / "mae_line_plot_dashed_is_explain_unhelpful.pdf", bbox_inches="tight")
    plt.show()

    lin_test = linregress(mae_df["mae_delta"], mae_df["explain_helpful"])
    print(f"Slope: {lin_test.slope:.3f}, r: {lin_test.rvalue:.3f}, p: {lin_test.pvalue:.3f}")

    # For abstract
    fig, ax = plot_mae_lines(explain_unhelpful_idx, legend=False, figsize=(10, 8))
    line1 = mlines.Line2D([], [], color='gray', linestyle="-",
                          label='Agree')
    line2 = mlines.Line2D([], [], color='gray', linestyle="--",
                          label='Disagree')
    legend = ax.legend(handles=[line1, line2], title="     I found the explanations \nhelpful in making my estimates.", fontsize=14, ncol=2)
    ax.set_ylim([9, 34])
    plt.setp(legend.get_title(), fontsize=14)
    plt.savefig(out_dir / "mae_line_plot_dashed_is_explain_unhelpful_abstract.png", bbox_inches="tight", dpi=300)
    plt.show()

    fig, ax = plt.subplots()
    ax.bar(["Stage 2", "Stage 3"],
           [df["pred_agrees_1"].sum(), df["pred_agrees_2"].sum()])
    plt.show()

    stage_1_agreement = df.groupby("study_id").pred_agrees_0.sum() / 65
    stage_2_agreement = df.groupby("study_id").pred_agrees_1.sum() / 65
    stage_3_agreement = df.groupby("study_id").pred_agrees_2.sum() / 65

    agreement_df = pd.DataFrame(
        {"Stage 1": stage_1_agreement, "Stage 2": stage_2_agreement, "Stage 3": stage_3_agreement})
    agreement_df = agreement_df.loc[study_id_order]
    plt.rcParams["font.size"] = 16
    bar_width = 0.35
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(agreement_df))
    rects1 = ax.bar(x - bar_width / 2, 100 * agreement_df['Stage 2'], bar_width,
                    label='Stage 2')
    rects2 = ax.bar(x + bar_width / 2, 100 * agreement_df['Stage 3'], bar_width,
                    label='Stage 3')
    ax.set_xlabel('Participant ID')
    ax.set_ylabel('Agreement with XAI / %')
    # ax.set_xticks(x, labels=agreement_df.index, rotation=45, ha="right")
    ax.set_xticks(x, labels=x)
    ax.legend(loc=(0.15, 0.75))
    plt.tight_layout()
    plt.savefig(out_dir / "participant_agreement.png", bbox_inches="tight")
    plt.show()

    plt.rcParams["font.size"] = 16
    bar_width = 0.25
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(agreement_df))
    rects0 = ax.bar(x - bar_width, 100 * agreement_df['Stage 1'], bar_width,
                    label='Stage 1')
    rects1 = ax.bar(x, 100 * agreement_df['Stage 2'], bar_width,
                    label='Stage 2')
    rects2 = ax.bar(x + bar_width, 100 * agreement_df['Stage 3'], bar_width,
                    label='Stage 3')
    ax.set_xlabel('Participant ID')
    ax.set_ylabel('Agreement with XAI / %')
    # ax.set_xticks(x, labels=agreement_df.index, rotation=45, ha="right")
    ax.set_xticks(x, labels=x)
    ax.legend(loc=(0.15, 0.70))
    plt.tight_layout()
    plt.savefig(out_dir / "participant_agreement_w_stage1.png", bbox_inches="tight")
    plt.show()

    plt.rcParams["font.size"] = 16
    stage_2_appropriate_reliance = df.groupby("study_id")["appropriate_reliance_1"].sum() / 65
    stage_3_appropriate_reliance = df.groupby("study_id")["appropriate_reliance_2"].sum() / 65
    reliance_df = pd.DataFrame(
        {"Stage 2": stage_2_appropriate_reliance, "Stage 3": stage_3_appropriate_reliance})
    reliance_df = reliance_df.loc[study_id_order]

    bar_width = 0.35
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(reliance_df))
    rects1 = ax.bar(x - bar_width / 2, 100 * reliance_df['Stage 2'], bar_width,
                    label='Stage 2')
    rects2 = ax.bar(x + bar_width / 2, 100 * reliance_df['Stage 3'], bar_width,
                    label='Stage 3')
    ax.set_xlabel('Participant ID')
    ax.set_ylabel('Appropriate Reliance on XAI / %')
    # ax.set_xticks(x, labels=agreement_df.index, rotation=45, ha="right")
    ax.set_xticks(x, labels=x)
    ax.legend(loc=(0.15, 0.75))
    plt.tight_layout()
    plt.savefig(out_dir / "participant_appropriate_reliance.png", bbox_inches="tight")
    plt.show()

    reliance_types = ["over_reliance", "under_reliance", "appropriate_reliance"]
    participant_reliance = {}
    for reliance_type in reliance_types:
        participant_reliance[reliance_type] = [
            100 * df.groupby("study_id")[f"{reliance_type}_1"].sum().loc[study_id_order] / 65,
            100 * df.groupby("study_id")[f"{reliance_type}_2"].sum().loc[study_id_order] / 65,
        ]

    def print_results(x1, x2, precision=2, f=ttest_ind):
        ttest = f(x1, x2)
        print(
            f"{x1.mean():.{precision}f}, "
            f"{x1.std():.{precision}f} SD"
            f" vs "
            f"{x2.mean():.{precision}f}, "
            f"{x2.std():.{precision}f} SD, "
            f"p={ttest.pvalue:.8f}"
        )
    for reliance_type in reliance_types:
        print(reliance_type)
        print_results(*participant_reliance[reliance_type])

    for i in range(2):
        print(f"Stage {i+2}")
        x1 = participant_reliance["appropriate_reliance"][i]
        x2 = participant_reliance["under_reliance"][i]
        print_results(x1, x2)
        log_o = np.log(x1 / x2)
        print_results(log_o, np.zeros_like(log_o), precision=2, f=ttest_rel)

    for i in range(2):
        print(f"Stage {i+2}")
        x1 = participant_reliance["under_reliance"][i]
        x2 = participant_reliance["over_reliance"][i]
        print_results(x1, x2)
        log_o = np.log(x1 / x2)
        print_results(log_o, np.zeros_like(log_o), precision=2, f=ttest_rel)

    print_results(agreement_df["Stage 1"], agreement_df["Stage 2"], 2)
    print_results(agreement_df["Stage 2"], agreement_df["Stage 3"], 2)


    handles = [
        mpatches.Patch(color="C0", label="Over Reliance"),
        mpatches.Patch(color="C1", label="Under Reliance"),
        mpatches.Patch(color="C2", label="Appropriate Reliance"),
    ]
    reliance_types_pretty = ["Over Reliance", "Under Reliance", "Appropriate Reliance"]
    stages = ["Stage 2", "Stage 3"]
    fig, ax = plt.subplots(figsize=(10, 5))
    bottom = np.zeros(10)
    positions = np.arange(10).astype(float)*2
    for reliance_type in reliance_types:
        ax.bar(
            np.arange(10),
            participant_reliance[reliance_type][0],
            label=reliance_type,
            bottom=bottom
        )
        bottom += participant_reliance[reliance_type][0].values

        positions += 0.40

    ax.set_xlabel("Participant ID")
    ax.set_xticks(np.arange(10))
    ax.set_ylabel("Proportion of images")
    ax.legend(handles=handles)
    plt.tight_layout()
    plt.show()

    fig, ax = plt.subplots(figsize=(10, 5))
    bottom = [np.zeros(10), np.zeros(10)]
    bar_width = 0.38
    colors = ["C0", "C1", "C2"]
    for i, reliance_type in enumerate(reliance_types):
        ax.bar(
            np.arange(10) - 0.21,
            participant_reliance[reliance_type][0],
            label=reliance_type,
            bottom=bottom[0],
            color=colors[i],
            width=bar_width,
        )
        ax.bar(
            np.arange(10) + 0.21,
            participant_reliance[reliance_type][1],
            label=reliance_type,
            bottom=bottom[1],
            color=colors[i],
            width=bar_width,
        )
        bottom[0] += participant_reliance[reliance_type][0].values
        bottom[1] += participant_reliance[reliance_type][1].values

    ax.set_xlabel("Participant ID")
    ax.set_xticks(np.arange(10))
    ax.set_ylabel("Proportion of images / %")
    ax.legend(handles=handles)
    plt.tight_layout()
    plt.savefig(out_dir / "participant_reliance.png", bbox_inches="tight")
    plt.show()

    explain_unhelpful_ids = mae_df.loc[mae_df["explain_helpful"] <= 3].study_id
    fig, ax = plt.subplots(figsize=(10, 5))
    bottom = [np.zeros(10), np.zeros(10)]
    bar_width = 0.38
    colors = ["C0", "C1", "C2"]
    for i, reliance_type in enumerate(reliance_types):
        ax.bar(
            np.arange(10) - 0.21,
            participant_reliance[reliance_type][0],
            label=reliance_type,
            bottom=bottom[0],
            color=colors[i],
            width=bar_width,
        )
        ax.bar(
            np.arange(10) + 0.21,
            participant_reliance[reliance_type][1],
            label=reliance_type,
            bottom=bottom[1],
            color=colors[i],
            width=bar_width,
        )
        bottom[0] += participant_reliance[reliance_type][0].values
        bottom[1] += participant_reliance[reliance_type][1].values

    ax.set_xlabel("Participant ID")
    ax.set_xticks(np.arange(10))
    ax.set_ylabel("Proportion of images / %")
    ax.legend(handles=handles)
    plt.tight_layout()
    plt.savefig(out_dir / "participant_reliance_by_helpful.png", bbox_inches="tight")
    plt.show()


    fig, ax = plt.subplots()
    sns.histplot(df.loc[df["appropriate_reliance_1"], "moves_closer_val_1"].abs(),
                 label="Appropriate reliance", ax=ax, color="C2",
                 bins=np.arange(0, df["moves_closer_val_1"].abs().max() + 1))
    sns.histplot(df.loc[df["over_reliance_1"], "moves_closer_val_1"].abs(), label="Over reliance", ax=ax, color="C0", bins=np.arange(0, df["moves_closer_val_1"].abs().max() + 1))
    sns.histplot(df.loc[df["under_reliance_1"], "moves_closer_val_1"].abs(), label="Under reliance", ax=ax, color="C1", bins=np.arange(0, df["moves_closer_val_1"].abs().max() + 1))
    plt.legend()
    plt.xlabel("Absolute change in participant prediction")
    plt.savefig(out_dir / "reliance_dists_1.pdf")

    fig, ax = plt.subplots()
    sns.histplot(df.loc[df["appropriate_reliance_2"], "moves_closer_val_2"].abs(),
                 label="Appropriate reliance", ax=ax, color="C2",
                 bins=np.arange(0, df["moves_closer_val_2"].abs().max() + 1))
    sns.histplot(df.loc[df["over_reliance_2"], "moves_closer_val_2"].abs(), label="Over reliance", ax=ax, color="C0", bins=np.arange(0, df["moves_closer_val_2"].abs().max() + 1))
    sns.histplot(df.loc[df["under_reliance_2"], "moves_closer_val_2"].abs(), label="Under reliance", ax=ax, color="C1", bins=np.arange(0, df["moves_closer_val_2"].abs().max() + 1))
    plt.legend()
    plt.xlabel("Absolute change in participant prediction")
    plt.savefig(out_dir / "reliance_dists_2.pdf")

    df["moves_closer_val_abs_1"] = df["moves_closer_val_1"].abs()
    df["moves_closer_val_abs_2"] = df["moves_closer_val_2"].abs()

    df['reliance_1'] = df.apply(lambda row:
                              'Over Reliance' if row['over_reliance_1'] == 1 else
                              'Under Reliance' if row['under_reliance_1'] == 1 else
                              'Appropriate Reliance', axis=1)

    df['reliance_2'] = df.apply(lambda row:
                              'Over Reliance' if row['over_reliance_2'] == 1 else
                              'Under Reliance' if row['under_reliance_2'] == 1 else
                              'Appropriate Reliance', axis=1)

    handles = [
        mpatches.Patch(color="C0", label="Over Reliance"),
        mpatches.Patch(color="C1", label="Under Reliance"),
        mpatches.Patch(color="C2", label="Appropriate Reliance"),
    ]
    sns.displot(
        df.reset_index(),
        x="moves_closer_val_abs_1",
        hue="reliance_1",
        hue_order=["Over Reliance", "Under Reliance", "Appropriate Reliance"],
        multiple="stack",
        legend=False,
    )
    plt.legend(handles=handles)
    plt.xlabel("Absolute change in participant estimate")
    plt.xlim(0, None)
    fig = plt.gcf()
    fig.set_size_inches(6, 4)
    plt.tight_layout()
    plt.savefig(out_dir / "reliance_dists_stage_2.pdf", bbox_inches="tight")
    plt.show()

    sns.displot(
        df.reset_index(),
        x="moves_closer_val_abs_2",
        hue="reliance_2",
        hue_order=["Over Reliance", "Under Reliance", "Appropriate Reliance"],
        multiple="stack",
        legend=False,
    )
    plt.legend(handles=handles)
    plt.xlabel("Absolute change in participant estimate")
    plt.xlim(0, None)
    fig = plt.gcf()
    fig.set_size_inches(6, 4)
    plt.tight_layout()
    plt.savefig(out_dir / "reliance_dists_stage_3.pdf", bbox_inches="tight")
    plt.show()

    df["exp0_to_exp1"] = df["exp_1"] - df["exp_0"]
    df["exp0_to_exp2"] = df["exp_2"] - df["exp_0"]
    df["exp0_to_exp1_abs"] = df["exp0_to_exp1"].abs()
    df["exp0_to_exp2_abs"] = df["exp0_to_exp2"].abs()

    handles = [
        mpatches.Patch(color="C0", label="Over Reliance"),
        mpatches.Patch(color="C1", label="Under Reliance"),
        mpatches.Patch(color="C2", label="Appropriate Reliance"),
    ]
    sns.displot(
        df.reset_index(),
        x="exp0_to_exp1_abs",
        hue="reliance_1",
        hue_order=["Over Reliance", "Under Reliance", "Appropriate Reliance"],
        multiple="stack",
        legend=False,
        bins=np.arange(df["exp0_to_exp1_abs"].max() + 2) - 0.5
    )
    plt.legend(handles=handles)
    plt.xlabel("Participant shift / weeks")
    plt.xlim(-0.5, None)
    xticks = plt.xticks(np.arange(0, df["exp0_to_exp1_abs"].max() + 1, 2))
    # for label in xticks[1][1::2]:
    #     label.set_visible(False)
    fig = plt.gcf()
    fig.set_size_inches(6, 4)
    plt.tight_layout()
    plt.savefig(out_dir / "shift_dists_stage_2.pdf", bbox_inches="tight")
    plt.show()

    print("Over/Under/Appropriate Reliance")
    print(df["over_reliance_1"].sum() / len(df), df["under_reliance_1"].sum() / len(df), df["appropriate_reliance_1"].sum() / len(df))
    print(df["over_reliance_2"].sum() / len(df), df["under_reliance_2"].sum() / len(df), df["appropriate_reliance_2"].sum() / len(df))

    for reliance_type in ["over_reliance", "under_reliance", "appropriate_reliance"]:
        for stage_minus_1 in [1, 2]:
            print("Movement towards model:", reliance_type, stage_minus_1 + 1, f'{df.loc[df[f"{reliance_type}_{stage_minus_1}"], f"moves_closer_val_abs_{stage_minus_1}"].mean():.2f}')
            print("Movement              :", reliance_type, stage_minus_1 + 1, f'{df.loc[df[f"{reliance_type}_{stage_minus_1}"], f"model_diff_{stage_minus_1}"].abs().mean():.2f}')

    for reliance_type in ["over_reliance", "under_reliance", "appropriate_reliance"]:
        for stage_minus_1 in [1, 2]:
            print("Movement towards model:", reliance_type, stage_minus_1 + 1, f'{df.loc[df[f"{reliance_type}_{stage_minus_1}"], f"moves_closer_val_abs_{stage_minus_1}"].mean()*7:.1f}')

    reliance_metrics_old = {
        "Stage 2": {
            "proportion": {
                "Over Reliance": df["over_reliance_1"].sum() / len(df),
                "Under Reliance": df["under_reliance_1"].sum() / len(df),
                "Appropriate Reliance": df["appropriate_reliance_1"].sum() / len(df),
            },
            "model shift_mean": {
                "Over Reliance": df.loc[df[f"over_reliance_1"], f"moves_closer_val_abs_1"].mean() * 7,
                "Under Reliance": df.loc[df[f"under_reliance_1"], f"moves_closer_val_abs_1"].mean() * 7,
                "Appropriate Reliance": df.loc[df[f"appropriate_reliance_1"], f"moves_closer_val_abs_1"].mean() * 7,
            },
            "model shift": {
                "Over Reliance": df.loc[df[
                    f"over_reliance_1"], f"moves_closer_val_abs_1"] * 7,
                "Under Reliance": df.loc[df[
                    f"under_reliance_1"], f"moves_closer_val_abs_1"] * 7,
                "Appropriate Reliance": df.loc[df[
                    f"appropriate_reliance_1"], f"moves_closer_val_abs_1"] * 7,
            }
        },
        "Stage 3": {
            "proportion": {
                "Over Reliance": df["over_reliance_2"].sum() / len(df),
                "Under Reliance": df["under_reliance_2"].sum() / len(df),
                "Appropriate Reliance": df["appropriate_reliance_2"].sum() / len(df),
            },
            "model shift_mean": {
                "Over Reliance": df.loc[df[f"over_reliance_2"], f"moves_closer_val_abs_2"].mean() * 7,
                "Under Reliance": df.loc[df[f"under_reliance_2"], f"moves_closer_val_abs_2"].mean() * 7,
                "Appropriate Reliance": df.loc[df[f"appropriate_reliance_2"], f"moves_closer_val_abs_2"].mean() * 7,
            },
            "model shift": {
                "Over Reliance": df.loc[df[f"over_reliance_2"], f"moves_closer_val_abs_2"] ,
                "Under Reliance": df.loc[df[f"under_reliance_2"], f"moves_closer_val_abs_2"],
                "Appropriate Reliance": df.loc[df[f"appropriate_reliance_2"], f"moves_closer_val_abs_2"],
            }
        },
    }


    reliance_metrics = {
        "proportion": {
            "Over Reliance": [df["over_reliance_1"].sum() / len(df), df["over_reliance_2"].sum() / len(df)],
            "Under Reliance": [df["under_reliance_1"].sum() / len(df), df["under_reliance_2"].sum() / len(df)],
            "Appropriate Reliance": [df["appropriate_reliance_1"].sum() / len(df), df["appropriate_reliance_2"].sum() / len(df)],
        },
        "model shift mean": {
            "Over Reliance": [
                df.loc[df[f"over_reliance_1"], f"moves_closer_val_abs_1"].mean() * 7,
                df.loc[df[f"over_reliance_2"], f"moves_closer_val_abs_2"].mean() * 7
            ],
            "Under Reliance": [
                df.loc[df[f"under_reliance_1"], f"moves_closer_val_abs_1"].mean() * 7,
                df.loc[df[f"under_reliance_2"], f"moves_closer_val_abs_2"].mean() * 7
            ],
            "Appropriate Reliance": [
                df.loc[df[f"appropriate_reliance_1"], f"moves_closer_val_abs_1"].mean() * 7,
                df.loc[df[f"appropriate_reliance_2"], f"moves_closer_val_abs_2"].mean() * 7
            ],
        },
        "model shift": {
            "Over Reliance": [
                df.loc[df[f"over_reliance_1"], f"moves_closer_val_abs_1"],
                df.loc[df[f"over_reliance_2"], f"moves_closer_val_abs_2"]
            ],
            "Under Reliance": [
                df.loc[df[f"under_reliance_1"], f"moves_closer_val_abs_1"],
                df.loc[df[f"under_reliance_2"], f"moves_closer_val_abs_2"]
            ],
            "Appropriate Reliance": [
                df.loc[df[f"appropriate_reliance_1"], f"moves_closer_val_abs_1"],
                df.loc[df[f"appropriate_reliance_2"], f"moves_closer_val_abs_2"]
            ],
        },
        "response shift": {
            "Over Reliance": [
                df.loc[df[f"over_reliance_1"], f"model_diff_1"].abs(),
                df.loc[df[f"over_reliance_2"], f"model_diff_2"].abs()
            ],
            "Under Reliance": [
                df.loc[df[f"under_reliance_1"], f"model_diff_1"].abs(),
                df.loc[df[f"under_reliance_2"], f"model_diff_2"].abs()
            ],
            "Appropriate Reliance": [
                df.loc[df[f"appropriate_reliance_1"], f"model_diff_1"].abs(),
                df.loc[df[f"appropriate_reliance_2"], f"model_diff_2"].abs()
            ],
        }
    }

    handles = [
        mpatches.Patch(color="C0", label="Over Reliance"),
        mpatches.Patch(color="C1", label="Under Reliance"),
        mpatches.Patch(color="C2", label="Appropriate Reliance"),
    ]
    reliance_types = ["Over Reliance", "Under Reliance", "Appropriate Reliance"]
    stages = ["Stage 2", "Stage 3"]
    fig, axes = plt.subplots(1, 2)
    bottom = np.zeros(2)
    positions = np.arange(2).astype(float)*2
    for reliance_type in reliance_types:
        axes[0].bar(
            stages,
            reliance_metrics["proportion"][reliance_type],
            label=reliance_type,
            bottom=bottom
        )
        bottom += np.array(reliance_metrics["proportion"][reliance_type])

        axes[1].violinplot(
            reliance_metrics["model shift"][reliance_type],
            positions=positions,
            showmeans=True,
        )
        positions += 0.40

    axes[1].yaxis.tick_right()
    axes[1].yaxis.set_label_position("right")
    axes[1].set_xticks(np.arange(2).astype(float)*2 + 0.40)
    axes[1].set_xticklabels(stages)
    axes[1].set_ylabel("Absolute participant shift towards model") # "Absolute change in estimate towards model prediction"
    axes[0].set_ylabel("Proportion of images")
    axes[1].legend(handles=handles)
    plt.show()

    fig, axes = plt.subplots(1, 2)
    bottom = np.zeros(2)
    positions = np.arange(2).astype(float) * 2
    for reliance_type in reliance_types:
        axes[0].bar(
            stages,
            reliance_metrics["proportion"][reliance_type],
            label=reliance_type,
            bottom=bottom
        )
        bottom += np.array(reliance_metrics["proportion"][reliance_type])

        axes[1].violinplot(
            reliance_metrics["response shift"][reliance_type],
            positions=positions,
            showmeans=True,
        )
        positions += 0.40

    axes[1].yaxis.tick_right()
    axes[1].yaxis.set_label_position("right")
    axes[1].set_xticks(np.arange(2).astype(float) * 2 + 0.40)
    axes[1].set_xticklabels(stages)
    axes[1].set_ylabel("Participant shift")
    axes[0].set_ylabel("Proportion of images")
    axes[1].legend(handles=handles)
    plt.show()

    def plot_mae_vs(feature, stage=3, xlabel=None):
        if xlabel is None:
            xlabel = feature
        fig, ax = plt.subplots()
        ax.scatter(mae_df[feature], mae_df[f"Stage {stage}"])
        ax.set_xlabel(xlabel)
        ax.set_ylabel(f"Stage {stage} MAE / days")
        plt.savefig(out_dir / f"mae_{stage}_vs_{feature}.png", bbox_inches="tight")

        fig, ax = plt.subplots()
        ax.scatter(stage_3_better_maes[feature], stage_3_better_maes[f"Stage {stage}"])
        ax.scatter(stage_3_worse_maes[feature], stage_3_worse_maes[f"Stage {stage}"])
        ax.set_xlabel(xlabel)
        ax.set_ylabel(f"Stage {stage} MAE / days")
        plt.savefig(out_dir / f"color_mae_{stage}_vs_{feature}.png", bbox_inches="tight")

    plot_mae_vs("explain_helpful", xlabel="Explanation was helpful")
    plot_mae_vs("familiar_heatmaps", xlabel="Familiar with heatmaps")
    plot_mae_vs("followed_2", xlabel="Followed algorithm's predictions")
    plot_mae_vs("trust_2", xlabel="Trust algorithm")
    plot_mae_vs("years_ultrasound", xlabel="Years experience")
    plot_mae_vs("trust_combo_2", xlabel="Trust score")
    plot_mae_vs("trust_combo", stage=2, xlabel="Trust score")

    mae_df["familiar_heatmaps_eq_1"] = mae_df["familiar_heatmaps"] == 1
    fig, ax = plot_mae_lines(np.where(~mae_df["familiar_heatmaps_eq_1"])[0])
    plt.savefig(out_dir / "mae_line_plot_by_familiar_heatmap.png", bbox_inches="tight")

    mae_df["gets_worse"] = mae_df["Stage 3"] > mae_df["Stage 2"]

    bins = np.arange(0, 1, 0.1)
    sns.distplot(mae_df["trust_combo"], kde=False, bins=bins)
    sns.distplot(mae_df["trust_combo_2"], kde=False, bins=bins)
    plt.show()

    trust_long = pd.melt(
        mae_df,
        id_vars=['study_id', "familiar_heatmaps_eq_1", "gets_worse"],
        value_vars=["trust_combo", "trust_combo_2"],
        var_name='Stage',
        value_name='Trust Score'
    )
    sns.displot(trust_long, x="Trust Score", row="Stage")
    plt.savefig(out_dir / "trust_score_dist.png", bbox_inches="tight")
    sns.displot(trust_long, x="Trust Score", row="Stage", hue="gets_worse")
    plt.savefig(out_dir / "trust_score_dist_by_score_change.png", bbox_inches="tight")

    mae_long = pd.melt(
        mae_df,
        id_vars=['study_id', "familiar_heatmaps_eq_1"],
        value_vars=['Stage 1', 'Stage 2', 'Stage 3'],
        var_name='Stage',
        value_name='MAE'
    )
    sns.displot(mae_long, x="MAE", row="Stage")
    plt.savefig(out_dir / "mae_dist.png", bbox_inches="tight")

    sns.displot(mae_long, x="MAE", row="Stage", hue="familiar_heatmaps_eq_1")
    plt.show()

    ids = mae_df.loc[mae_df["familiar_heatmaps"] > 1].study_id.values
    sns.displot(mae_long.loc[~mae_long.study_id.isin(ids)], x="MAE", row="Stage")
    sns.displot(mae_long.loc[mae_long.study_id.isin(ids)], x="MAE", row="Stage")
    plt.show()

    explain_not_helpful = np.where(mae_df.explain_helpful < 3)[0]
    fig, ax = plot_mae_lines(explain_not_helpful)
    plt.savefig(out_dir / "mae_line_plot_dashed_is_explain_not_helpful.png", bbox_inches="tight")

    trust_neutral = np.where(mae_df.ml_trust == 3)[0]
    fig, ax = plot_mae_lines(trust_neutral)
    plt.savefig(out_dir / "mae_line_plot_dashed_is_trust_neutral.png", bbox_inches="tight")

    not_used_in_clinical = np.where(mae_df.ml_clinical_practice < 3)[0]
    fig, ax = plot_mae_lines(not_used_in_clinical)
    plt.savefig(out_dir / "mae_line_plot_dashed_is_ml_not_used_in_clinical.png",
                bbox_inches="tight")

    mae_df.to_csv(base_dir / "results.csv", index=False)

    print("Done!")


def process_jisc_stage_1(df):
    rename_ids_dict = {
        "01001": "AGE-0c1312af",
        "01002": "AGE-0599ed56",
        "01005": "AGE-3f5188ec",
        "AGE-3f5188ec": "AGE-b459bccb",
    }
    rename_cols = {
        "1. Name": "study_id",
        "2. Your age range (in years)?": "age",
        '3. Which region of the UK do you work in?': "region",
        "4. Which of these most closely matches your job title?": "job_title",
        "5. How many years have you been performing fetal ultrasound for?": "years_ultrasound",
        "6. How often do you perform obstetric scans?": "ultrasound_freq",
        '7.1.a. I am comfortable using new technology': "new_tech",
        '7.2.a. I understand the meaning of the term machine learning': "ml_meaning",
        '7.3.a. I am currently using machine learning-based algorithms to aid my clinical practice': "ml_clinical_practice",
        '7.4.a. I would be comfortable incorporating machine learning-based algorithms into my clinical practice': "ml_comfort",
        '7.5.a. I would be comfortable incorporating machine learning-based algorithms for gestational age estimation into my clinical practice': "ml_comfort_ga",
        '7.6.a. I trust machine learning-based algorithms': "ml_trust",
        '7.7.a. I distrust machine learning-based algorithms': "ml_distrust",
        '7.8.a. I am familiar with using image heatmaps for image analysis': "familiar_heatmaps",
    }
    df = df.rename(columns=rename_cols)
    df.loc[:, "study_id"] = df.loc[:, "study_id"].replace(rename_ids_dict)
    df = df.dropna(axis=1, how="all")

    num_cols = ['new_tech', 'ml_meaning', 'ml_clinical_practice', 'ml_comfort', 'ml_comfort_ga', 'ml_trust', 'ml_distrust', 'familiar_heatmaps']
    df = convert_string_answers_to_int(df, num_cols)
    return df


def process_jisc_stage_2(df):
    rename_cols = {
        '1. Name': 'study_id',
        '2.1.a. The algorithm’s estimate was helpful in my decision making': 'helpful',
        '2.2.a. I felt the algorithm’s estimates were accurate': 'accurate',
        '2.3.a. I followed the algorithm’s recommendations every time': 'followed',
        '2.4.a. I would be comfortable incorporating the algorithm into my clinical practice': 'clinical_practice',
        '2.5.a. I trust the algorithm': 'trust',
        '2.6.a. I distrust the algorithm': 'distrust',
        '3.1.a. I am confident in the algorithm. I feel that it works well': 'confident',
        '3.2.a. The outputs of the algorithm are very predictable': 'predictable',
        '3.3.a. The algorithm is very reliable. I can count on it to be correct all the time': 'reliable',
        '3.4.a. I feel safe that when I rely on the algorithm I will get the right answers': 'safe_to_rely',
        '3.5.a. I am wary of the algorithm': 'wary',
        '3.6.a. I like using the system for gestational age estimation': 'like'
    }

    df = df.rename(columns=rename_cols)
    df = df.dropna(axis=1, how="all")

    num_cols = ['helpful', 'accurate', 'followed', 'clinical_practice', 'trust', 'distrust', 'confident', 'predictable', 'reliable', 'safe_to_rely', 'wary', 'like']
    df = convert_string_answers_to_int(df, num_cols)
    return df


def process_jisc_stage_3(df):
    rename_cols = {
        '1. Name': 'study_id',
        '2.1.a. The algorithm’s estimate was helpful in my decision making': 'helpful',
        '2.2.a. I felt the algorithm’s estimates were accurate': 'accurate',
        '2.3.a. I followed the algorithm’s recommendations every time': 'followed',
        '2.4.a. I would be comfortable incorporating the algorithm into my clinical practice': 'clinical_practice',
        '2.5.a. I trust the algorithm': 'trust',
        '2.6.a. I distrust the algorithm': 'distrust',
        '2.7.a. I found the explanations helpful in making my estimates': 'explain_helpful',
        '2.8.a. I found the explanations helpful for all of the images': 'explain_helpful_all',
        '2.9.a. I found the explanations increased my level of trust in the model’s estimate': 'explain_trust',
        '3.1.a. I am confident in the algorithm. I feel that it works well': 'confident',
        '3.2.a. The outputs of the algorithm are very predictable': 'predictable',
        '3.3.a. The algorithm is very reliable. I can count on it to be correct all the time': 'reliable',
        '3.4.a. I feel safe that when I rely on the algorithm I will get the right answers': 'safe_to_rely',
        '3.5.a. I am wary of the algorithm': 'wary',
        '3.6.a. I like using the system for gestational age estimation': 'like',
        '4.1.a. Providing the explanations would be useful for clinical decision making': 'explain_clinical',
        '4.2.a. I found the explanations interesting': 'explain_interesting',
        '4.3.a. I would feel confident using a machine learning-based algorithm without any explanation': 'confident_no_explaination',
        '4.4.a. I would feel confident using a machine learning-based algorithm where no-one (clinicians or engineers) could understand how it reached its estimate as long as it had been through rigorous testing': 'confident_no_explaination_detail',
        '5.1.a. I would be comfortable incorporating machine learning-based algorithms into my clinical practice': 'ml_comfort',
        '5.2.a. I would be comfortable incorporating machine learning-based algorithms for gestational age estimation into my clinical practice': 'ml_comfort_ga',
        '5.3.a. I trust machine learning-based algorithms': 'ml_trust',
        '5.4.a. I distrust machine learning-based algorithms': 'ml_distrust',
        '6. In what way did the explanations provided in stage 3 influence your decision-making?': 'explanations_free_text',
        '7. Did you find information outside of the foetal head important in making your decision?': 'outside_head_free_text'
    }

    df = df.rename(columns=rename_cols)
    df = df.dropna(axis=1, how="all")

    num_cols = list(df.columns.values)
    num_cols = num_cols[num_cols.index("study_id")+1:-2]
    df = convert_string_answers_to_int(df, num_cols)
    return df


def convert_string_answers_to_int(df, num_cols):
    for col in num_cols:
        df[col] = df.loc[:, col].str.extract(r'\((\d+)\)').astype(int)
    return df


def load_csv(path, study_id):
    df = pd.read_csv(path)
    df["study_id"] = study_id
    return df


if __name__ == "__main__":
    main()
