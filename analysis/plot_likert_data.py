
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import plot_likert
import numpy as np
from matplotlib import ticker
from scipy.stats import ttest_ind, linregress, mannwhitneyu, wilcoxon, chi2


class Result:
    def __init__(self, chi2_stat, pvalue):
        self.chi2_stat = chi2_stat
        self.pvalue = pvalue


def main():
    data_path = Path("results/results.csv")
    out_dir = Path("results/likert")
    df = pd.read_csv(data_path)

    question = "On a scale of 1-5, how much do you agree with the following statements?"
    scale = ["Strongly disagree (1)", "Somewhat disagree (2)", "Neutral (3)",
             "Somewhat agree (4)", "Strongly agree (5)"]
    num_scale = [1, 2, 3, 4, 5]

    helpful = df["explain_helpful"] >= 3
    not_helpful = df["explain_helpful"] < 3

    trust_columns = ["trust", "confident", "predictable", "reliable", "safe_to_rely", "like", "distrust", "wary"]
    trust_columns_2 = [v + "_2" for v in trust_columns]
   
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

    print_results(df["trust"], df["trust_2"], 2, mannwhitneyu)
    print_results(df["wary"], df["wary_2"], 2, mannwhitneyu)
    print_results(df["safe_to_rely"], df["safe_to_rely_2"], 2, mannwhitneyu)

    for col in trust_columns:
        print(col)
        print_results(df[col], df[col + "_2"], 2, wilcoxon)
        print((df[col + "_2"] - df[col]).median())
        print((df[col + "_2"] - df[col]).mean())

    col = "ml_comfort_ga"
    print_results(df[col], df[col + "_2"], 2, wilcoxon)
    print((df[col + "_2"] - df[col]).median())
    print((df[col + "_2"] - df[col]).mean())

    print_results(
        df.loc[:, trust_columns[:-2]].mean(axis=1),
        df.loc[:, trust_columns_2[:-2]].mean(axis=1),
        2,
        wilcoxon,
    )

    trust_df = pd.concat([pd.melt(df.loc[:, trust_columns[:-2]], value_name="Stage 2"), pd.melt(df.loc[:, trust_columns_2[:-2]], var_name="remove", value_name="Stage 3")], axis=1)
    trust_df.drop("remove", axis=1, inplace=True)
    print_results(trust_df["Stage 2"], trust_df["Stage 3"], 2, ttest_ind)
    print_results(trust_df["Stage 2"], trust_df["Stage 3"], 2, mannwhitneyu)
    print_results(trust_df["Stage 2"], trust_df["Stage 3"], 2, wilcoxon)
    print(trust_df["Stage 2"].median(), trust_df["Stage 3"].median())

    trust_df = pd.concat([pd.melt(df.loc[:, trust_columns[-2:]], value_name="Stage 2"), pd.melt(df.loc[:, trust_columns_2[-2:]], var_name="remove", value_name="Stage 3")], axis=1)
    trust_df.drop("remove", axis=1, inplace=True)
    print_results(trust_df["Stage 2"], trust_df["Stage 3"], 2, ttest_ind)
    print_results(trust_df["Stage 2"], trust_df["Stage 3"], 2, mannwhitneyu)
    print_results(trust_df["Stage 2"], trust_df["Stage 3"], 2, wilcoxon)

    for i in num_scale:
        df = df.applymap(lambda x: scale[i-1] if i == x else x)

    prior_to_study = ["ml_meaning", "ml_clinical_practice", "new_tech",
                      "familiar_heatmaps", "ml_comfort", "ml_comfort_ga", "ml_trust",
                      "ml_distrust"]

    post_study = ["ml_comfort_2", "ml_comfort_ga_2", "ml_trust_2", "ml_distrust_2"]
    explain_cols = ["explain_helpful", "explain_helpful_all", "explain_trust",
                    "explain_clinical", "explain_interesting"]
    explain_opinions = ["confident_no_explaination", "confident_no_explaination_detail"]

    fig, axes = plt.subplots(2, 1,)
    axes[0] = plot_with_questions(
        df.loc[helpful],
        stage_1_keys=prior_to_study,
        return_axes=True, label_max_width=60,
        figsize=(12, 6), bar_labels=True, ax=axes[0])
    axes[1] = plot_with_questions(
        df.loc[not_helpful],
        stage_1_keys=prior_to_study,
        return_axes=True, label_max_width=60,
        figsize=(12, 6), bar_labels=True, ax=axes[1])
    axes[1].legend_.remove()
    #axes[0].legend_.remove()#(bbox_to_anchor=(-0.30, 0.20))
    axes[0].figure.set_size_inches(14, 7, forward=True)
    plt.tight_layout()
    plt.savefig(out_dir / "explain_helpful_vs_not_likert.png")
    plt.show()

    plot_with_questions(
        df,
        stage_3_keys=explain_cols,
        savefig=out_dir / "explain_likert.png", label_max_width=55,
        figsize=(12, 4), bar_labels=True)

    plot_with_questions(
        df,
        stage_3_keys=explain_opinions,
        savefig=out_dir / "explain_opinions_likert.pdf", label_max_width=55,
        figsize=(12, 2), bar_labels=True)

    ax = plot_with_questions(
        df,
        stage_3_keys=explain_cols + explain_opinions,
        savefig=out_dir / "explain_and_opinions_likert.pdf",
        label_max_width=55, set_max_width=14,
        figsize=(12, 6), bar_labels=True, return_axes=True)
    ax.set_xlim([-0.75, 13.75])

    plt.tight_layout()
    plt.savefig(out_dir / "explain_and_opinions_likert.png", bbox_inches="tight")
    plt.show()

    fig, ax = plt.subplots(1, 1, )
    ax = plot_with_questions(
        df,
        stage_3_keys=explain_cols + explain_opinions,
        savefig=out_dir / "explain_and_opinions_likert.pdf",
        label_max_width=55, set_max_width=14,
        figsize=(12, 6), bar_labels=True, ax=ax, return_axes=True)
    ax.set_xlim([-0.75, 13.75])
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.05), ncol=5)
    ax.legend_.remove()
    plt.tight_layout()
    plt.savefig(out_dir / "explain_and_opinions_likert_paper.png", bbox_inches="tight")
    plt.show()

    fig, axes = plt.subplots(2, 1, height_ratios=[8, 4])
    axes[0] = plot_with_questions(
        df,
        stage_1_keys=prior_to_study,
        return_axes=True, label_max_width=55, horizontal_line_idx=0.5,
        figsize=(12, 6), bar_labels=True, ax=axes[0], set_max_width=20)
    axes[1] = plot_with_questions(
        df,
        stage_3_keys=post_study,
        return_axes=True, label_max_width=55, horizontal_line_idx=0.5,
        figsize=(12, 6), bar_labels=True, ax=axes[1], set_max_width=21, add_padding=3,)
    # xticks = np.arange(0, 21)
    # xlabels = list(range(10, 0, -1)) + list(range(0, 10 + 1))
    # axes[0].set_xticks(xticks)
    # axes[0].set_xticklabels(xlabels)
    # axes[1].set_xticks(xticks)
    # axes[1].set_xticklabels(xlabels)
    axes[0].figure.set_size_inches(14, 7, forward=True)
    axes[1].legend_.remove()
    plt.tight_layout()
    plt.savefig(out_dir / "prior_and_post_study_likert.png", bbox_inches="tight")
    plt.show()

    plot_with_questions(
        df,
        stage_2_keys=["trust"], stage_3_keys=["trust_2"],
        include_stage_titles=True, savefig=out_dir / "trust_likert.png",
        figsize=(9, 7), bar_labels=True
    )

    ax = plot_with_questions(
        df,
        stage_1_keys=prior_to_study,
        return_axes=True, label_max_width=67,
        figsize=(14, 5), bar_labels=True)
    plt.tight_layout()
    plt.savefig(out_dir / "prior_to_study_likert.png")
    plt.show()

    plot_with_questions(
        df,
        stage_3_keys=post_study, label_max_width=67,
        savefig=out_dir / "post_study_likert.png",
        figsize=(14, 5), bar_labels=True)

    plot_with_questions(
        df,
        stage_2_keys=trust_columns,
        savefig=out_dir / "trust_stage_2_likert.png", horizontal_line_idx=1.5,
        figsize=(9, 7), bar_labels=True)

    plot_with_questions(
        df,
        stage_3_keys=trust_columns_2,
        savefig=out_dir / "trust_stage_3_likert.png", horizontal_line_idx=1.5,
        figsize=(9, 7), bar_labels=True)


    fig, axes = plt.subplots(1, 2,)
    axes[0] = plot_with_questions(
        df,
        stage_2_keys=trust_columns,
        return_axes=True, horizontal_line_idx=1.5,
        figsize=(12, 6), bar_labels=True, ax=axes[0])
    axes[1] = plot_with_questions(
        df,
        stage_3_keys=trust_columns_2,
        return_axes=True, horizontal_line_idx=1.5,
        figsize=(12, 6), bar_labels=True, ax=axes[1])
    axes[1].set_yticks([])
    axes[0].legend_.remove()
    axes[0].set_title("Stage 2")
    axes[1].set_title("Stage 3")
    plt.tight_layout()
    plt.savefig(out_dir / "trust_both_stages_likert.png")
    plt.show()

    plt.rcParams['font.size'] = 12
    fig, axes = plt.subplots(2, 1,)
    axes[0] = plot_with_questions(
        df,
        stage_2_keys=trust_columns,
        return_axes=True, horizontal_line_idx=1.5, label_max_width=60,
        figsize=(12, 6), bar_labels=True, ax=axes[0])
    axes[1] = plot_with_questions(
        df,
        stage_3_keys=trust_columns_2,
        return_axes=True, horizontal_line_idx=1.5, label_max_width=60,
        figsize=(12, 6), bar_labels=True, ax=axes[1])
    axes[1].legend_.remove()
    #axes[0].legend_.remove()#(bbox_to_anchor=(-0.30, 0.20))
    axes[0].figure.set_size_inches(14, 7, forward=True)
    plt.tight_layout()
    plt.savefig(out_dir / "trust_both_stages_likert_vert.pdf")
    plt.show()

    fig, axes = plt.subplots(2, 1,)
    axes[0] = plot_with_questions(
        df,
        stage_2_keys=trust_columns,
        return_axes=True, horizontal_line_idx=1.5, label_max_width=60,
        figsize=(12, 6), bar_labels=True, ax=axes[0])
    axes[1] = plot_with_questions(
        df,
        stage_3_keys=trust_columns_2,
        return_axes=True, horizontal_line_idx=1.5, label_max_width=60,
        figsize=(12, 6), bar_labels=True, ax=axes[1], legend=False)
    axes[0].figure.set_size_inches(14, 7.5, forward=True)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.05), ncol=5)
    axes[0].legend_.remove()
    plt.tight_layout()
    plt.savefig(out_dir / "trust_both_stages_likert_paper.pdf", bbox_inches="tight")
    plt.show()

    trust_columns_subset = ["trust", "safe_to_rely", "distrust", "wary"]
    trust_columns_subset_2 = [v + "_2" for v in trust_columns_subset]
    plt.rcParams['font.size'] = 14
    fig, axes = plt.subplots(2, 1,)
    axes[0] = plot_with_questions(
        df,
        stage_2_keys=trust_columns_subset,
        return_axes=True, horizontal_line_idx=1.5, label_max_width=30,
        set_max_width=15, add_padding=0.5,
        figsize=(12, 6), bar_labels=True, ax=axes[0])
    axes[1] = plot_with_questions(
        df,
        stage_3_keys=trust_columns_subset_2,
        return_axes=True, horizontal_line_idx=1.5, label_max_width=30,
        set_max_width=15, add_padding=0.5,
        figsize=(12, 6), bar_labels=True, ax=axes[1])
    axes[0].set_xlim(-0.2, 15.2)
    axes[1].set_xlim(-0.2, 15.2)
    axes[1].legend_.remove()
    axes[0].legend(bbox_to_anchor=(0.25, 1.05)) #-0.30, 0.20
    axes[0].figure.set_size_inches(9, 9, forward=True)
    plt.tight_layout()
    plt.savefig(out_dir / "trust_both_stages_likert_vert_abstract.png", bbox_inches="tight", dpi=200)
    plt.show()


    participant_dirs = [Path("results") / v for v in df.study_id]
    confidence_data = []
    cols = [f"confidence_{i}" for i in range(3)]
    for participant_dir in participant_dirs:
        if participant_dir.is_dir():
            sub_data = pd.read_csv(participant_dir / "study_results.csv")
            confidence_data.append(sub_data[cols])

    confidence_df = pd.concat(confidence_data)
    confidence_df = confidence_df.fillna(2)
    confidence_df = confidence_df + 1
    confidence_df = confidence_df.rename(columns={"confidence_0": "Stage 1", "confidence_1": "Stage 2", "confidence_2": "Stage 3"})
    confidence_df = confidence_df.reset_index(drop=True)

    def assign_group(x):
        group_num = x // 65
        return df.study_id[group_num]

    confidence_df['study_id'] = confidence_df.index.map(assign_group)

    helpful_ids = df.loc[helpful, "study_id"]
    confidence_df["helpful"] = confidence_df["study_id"].isin(helpful_ids)

    stage_cols = ["Stage 1", "Stage 2", "Stage 3"]
    plot_likert.plot_likert(confidence_df.loc[:, stage_cols], [1, 2, 3, 4, 5], bar_labels=True, figsize=(6, 7))
    plt.title("Confidence")
    plt.tight_layout()
    plt.savefig(out_dir / "confidence_likert.png")
    plt.show()

    plot_likert.plot_likert(
        confidence_df.loc[confidence_df.helpful, stage_cols],[1, 2, 3, 4, 5],
        set_max_width=340, add_padding=94,
        bar_labels=True, figsize=(6, 7),
    )
    plt.title("Confidence")
    plt.tight_layout()
    plt.savefig(out_dir / "confidence_likert_helpful.png", bbox_inches="tight")
    plt.show()


    plot_likert.plot_likert(
        confidence_df.loc[~confidence_df.helpful, stage_cols], [1, 2, 3, 4, 5],
        set_max_width=500, add_padding=94,
        bar_labels=True, figsize=(6, 7))
    plt.title("Confidence")
    plt.tight_layout()
    plt.savefig(out_dir / "confidence_likert_not_helpful.png")
    plt.show()

    fig, axes = plt.subplots(2, 1)
    plot_likert.plot_likert(
        confidence_df.loc[confidence_df.helpful, stage_cols],[1, 2, 3, 4, 5],
        set_max_width=340, add_padding=94,
        bar_labels=True, figsize=(7, 5),
        ax=axes[0]
    )
    plot_likert.plot_likert(
        confidence_df.loc[~confidence_df.helpful, stage_cols], [1, 2, 3, 4, 5],
        set_max_width=500, add_padding=94,
        bar_labels=True, figsize=(7, 5),
        ax=axes[1], legend=False,
    )
    axes[0].set_title("Explanations Helpful")
    axes[1].set_title("Explanations Not Helpful")
    plt.tight_layout()
    plt.savefig(out_dir / "confidence_by_helpful_explanations.pdf", bbox_inches="tight")
    plt.show()

    fig, axes = plt.subplots(2, 1)
    plot_likert.plot_likert(
        confidence_df.loc[confidence_df.helpful, stage_cols],[1, 2, 3, 4, 5],
        set_max_width=340, add_padding=94,
        bar_labels=True, figsize=(7, 5),
        ax=axes[0]
    )
    plot_likert.plot_likert(
        confidence_df.loc[~confidence_df.helpful, stage_cols], [1, 2, 3, 4, 5],
        set_max_width=500, add_padding=94,
        bar_labels=True, figsize=(7, 5),
        ax=axes[1], legend=False,
    )
    plt.tight_layout()
    plt.savefig(out_dir / "confidence_by_helpful_explanations_paper.pdf", bbox_inches="tight")
    plt.show()

    print("Done!")


questions_stage_1 = {
    'age': 'Your age range (in years)?',
    'region': 'Which region of the UK do you work in?',
    'job_title': 'Which of these most closely matches your job title?',
    'years_ultrasound': 'How many years have you been performing fetal ultrasound for?',
    'ultrasound_freq': 'How often do you perform obstetric scans?',
    'new_tech': 'I am comfortable using new technology',
    'ml_meaning': 'I understand the meaning of the term machine learning',
    'ml_clinical_practice': 'I am currently using machine learning-based algorithms to aid my clinical practice',
    'ml_comfort': 'I would be comfortable incorporating machine learning-based algorithms into my clinical practice',
    'ml_comfort_ga': 'I would be comfortable incorporating machine learning-based algorithms for gestational age estimation into my clinical practice',
    'ml_trust': 'I trust machine learning-based algorithms',
    'ml_distrust': 'I distrust machine learning-based algorithms',
    'familiar_heatmaps': 'I am familiar with using image heatmaps for image analysis'
}

questions_stage_2 = {
    'helpful': 'The algorithm’s estimate was helpful in my decision making',
    'accurate': 'I felt the algorithm’s estimates were accurate',
    'followed': 'I followed the algorithm’s recommendations every time',
    'clinical_practice': 'I would be comfortable incorporating the algorithm into my clinical practice',
    'trust': 'I trust the algorithm',
    'distrust': 'I distrust the algorithm',
    'confident': 'I am confident in the algorithm. I feel that it works well',
    'predictable': 'The outputs of the algorithm are very predictable',
    'reliable': 'The algorithm is very reliable. I can count on it to be correct all the time',
    'safe_to_rely': 'I feel safe that when I rely on the algorithm I will get the right answers',
    'wary': 'I am wary of the algorithm',
    'like': 'I like using the system for gestational age estimation'
}

questions_stage_3 = {
    'helpful_2': 'The algorithm’s estimate was helpful in my decision making',
    'accurate_2': 'I felt the algorithm’s estimates were accurate',
    'followed_2': 'I followed the algorithm’s recommendations every time',
    'clinical_practice_2': 'I would be comfortable incorporating the algorithm into my clinical practice',
    'trust_2': 'I trust the algorithm',
    'distrust_2': 'I distrust the algorithm',
    'explain_helpful': 'I found the explanations helpful in making my estimates',
    'explain_helpful_all': 'I found the explanations helpful for all of the images',
    'explain_trust': 'I found the explanations increased my level of trust in the model’s estimate',
    'confident_2': 'I am confident in the algorithm. I feel that it works well',
    'predictable_2': 'The outputs of the algorithm are very predictable',
    'reliable_2': 'The algorithm is very reliable. I can count on it to be correct all the time',
    'safe_to_rely_2': 'I feel safe that when I rely on the algorithm I will get the right answers',
    'wary_2': 'I am wary of the algorithm',
    'like_2': 'I like using the system for gestational age estimation',
    'explain_clinical': 'Providing the explanations would be useful for clinical decision making',
    'explain_interesting': 'I found the explanations interesting',
    'confident_no_explaination': 'I would feel confident using a machine learning-based algorithm without any explanation',
    'confident_no_explaination_detail': 'I would feel confident using a machine learning-based algorithm where no-one (clinicians or engineers) could understand how it reached its estimate as long as it had been through rigorous testing',
    'ml_comfort_2': 'I would be comfortable incorporating machine learning-based algorithms into my clinical practice',
    'ml_comfort_ga_2': 'I would be comfortable incorporating machine learning-based algorithms for gestational age estimation into my clinical practice',
    'ml_trust_2': 'I trust machine learning-based algorithms',
    'ml_distrust_2': 'I distrust machine learning-based algorithms'
}

questions_stage_3_old = {
    'helpful': 'The algorithm’s estimate was helpful in my decision making',
    'accurate': 'I felt the algorithm’s estimates were accurate',
    'followed': 'I followed the algorithm’s recommendations every time',
    'clinical_practice': 'I would be comfortable incorporating the algorithm into my clinical practice',
    'trust': 'I trust the algorithm',
    'distrust': 'I distrust the algorithm',
    'explain_helpful': 'I found the explanations helpful in making my estimates',
    'explain_helpful_all': 'I found the explanations helpful for all of the images',
    'explain_trust': 'I found the explanations increased my level of trust in the model’s estimate',
    'confident': 'I am confident in the algorithm. I feel that it works well',
    'predictable': 'The outputs of the algorithm are very predictable',
    'reliable': 'The algorithm is very reliable. I can count on it to be correct all the time',
    'safe_to_rely': 'I feel safe that when I rely on the algorithm I will get the right answers',
    'wary': 'I am wary of the algorithm',
    'like': 'I like using the system for gestational age estimation',
    'explain_clinical': 'Providing the explanations would be useful for clinical decision making',
    'explain_interesting': 'I found the explanations interesting',
    'confident_no_explaination': 'I would feel confident using a machine learning-based algorithm without any explanation',
    'confident_no_explaination_detail': 'I would feel confident using a machine learning-based algorithm where no-one (clinicians or engineers) could understand how it reached its estimate as long as it had been through rigorous testing',
    'ml_comfort': 'I would be comfortable incorporating machine learning-based algorithms into my clinical practice',
    'ml_comfort_ga': 'I would be comfortable incorporating machine learning-based algorithms for gestational age estimation into my clinical practice',
    'ml_trust': 'I trust machine learning-based algorithms',
    'ml_distrust': 'I distrust machine learning-based algorithms'
}


def plot_with_questions(df, stage_1_keys=[], stage_2_keys=[], stage_3_keys=[], include_stage_titles=False, savefig=None, horizontal_line_idx=None, return_axes=False, **kwargs):
    if include_stage_titles:
        questions = (["Stage 1 \n" + questions_stage_1[k] for k in stage_1_keys]
                     + ["Stage 2 \n" + questions_stage_2[k] for k in stage_2_keys]
                     + ["Stage 3 \n" + questions_stage_3[k] for k in stage_3_keys])
    else:
        questions = ([questions_stage_1[k] for k in stage_1_keys]
                     + [questions_stage_2[k] for k in stage_2_keys]
                     + [questions_stage_3[k] for k in stage_3_keys])
    keys = stage_1_keys + stage_2_keys + stage_3_keys
    df = df.loc[:, keys]
    df.columns = questions
    scale = ["Strongly disagree (1)", "Somewhat disagree (2)", "Neutral (3)",
             "Somewhat agree (4)", "Strongly agree (5)"]

    ax = plot_likert.plot_likert(df, scale, **kwargs)

    if horizontal_line_idx is not None:
        if type(horizontal_line_idx) == int or type(horizontal_line_idx) == float:
            ax.axhline(horizontal_line_idx, color="gray")
        else:
            for idx in horizontal_line_idx:
                ax.axhline(idx, color="gray")

    if return_axes:
        return ax
    else:
        plt.tight_layout()
        if savefig:
            plt.savefig(savefig, bbox_inches="tight")
        plt.show()


if __name__ == '__main__':
    main()
