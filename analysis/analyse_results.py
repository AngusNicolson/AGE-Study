
from datetime import datetime
import json
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import seaborn as sns
import numpy as np
import pandas as pd
from scipy.stats import linregress, ttest_ind
from argparse import ArgumentParser

import cv2

class_to_age = {i: i*2 + 15 for i in range(13)}
class_to_age[0] = 14.5
class_to_age[12] = 40


def main(args):
    study_dir = Path().cwd()
    study_id = args.study_id
    orig_data_dir = study_dir / "data"
    study_data_dir = study_dir / "study_data"

    out_dir = study_dir / f"results/{study_id}"
    out_dir.mkdir(exist_ok=True)

    csv_paths = [
        orig_data_dir / f"stage_{i+1}/img_indices.csv" for i in range(3)
    ]

    stage_data_dirs = [study_data_dir / f"stage_{i+1}" for i in range(3)]
    json_paths = [list(stage_dir.glob("*.json")) for stage_dir in stage_data_dirs]
    study_ids_per_stage = [[path.name.split("_")[0] for path in paths] for paths in json_paths]

    print("No. participants to complete each stage")
    for i in range(3):
        print(f"Stage {i+1}: {len(set(study_ids_per_stage[i]))}")

    print(f"Analysing participant {study_id}")

    json_paths = [[path for path in paths if study_id in path.name] for paths in json_paths]
    completed_each_stage = [len(paths) == 1 for paths in json_paths]
    if not all(completed_each_stage):
        raise FileNotFoundError(f"Not all stages complete for participant {study_id}!")

    json_paths = [v[0] for v in json_paths]

    dfs = []
    for csv_path in csv_paths:
        df = pd.read_csv(csv_path)
        df = df.sort_values("out_idx")
        df = df.reset_index()
        df = df.rename(columns={"index": "in_idx"})
        df["age_pred"] = df.prediction.map(class_to_age)
        df["pred_error"] = df["age_pred"] - df["age"]
        dfs.append(df)

    data = [load_and_extract_jsons(path, True) if i == 0 else load_and_extract_jsons(path) for i, path in enumerate(json_paths)]
    n_samples = 1000
    maes = np.zeros((4, n_samples))
    for i, exp_data in enumerate(data):
        dfs[i][f"exp"] = exp_data["ages"]
        dfs[i][f"exp"] = dfs[i][f"exp"] + 13
        dfs[i]["confidence"] = exp_data["confidence"]
        dfs[i][f"exp_error"] = dfs[i][f"exp"] - dfs[i]["age"]
        dfs[i][f"exp_times"] = exp_data["times"]
        maes[i] = bootstrap_mae(
            dfs[i]["exp"], dfs[i]["age"],
            rng=np.random.default_rng(42), n_samples=n_samples, sample_size=0.6
        )
    dfs[0]["features"] = data[0]["features"]
    dfs[0]["features"] = dfs[0]["features"].str.split(",")
    dfs[0]["comment"] = data[0]["comment"]

    maes[3] = bootstrap_mae(
        dfs[0]["age_pred"], dfs[0]["age"],
        rng=np.random.default_rng(42), n_samples=n_samples, sample_size=0.6
    )

    merge_cols = ["path", "exp", "exp_error", "confidence"]
    cols = ["path", "target", "prediction", "age", "patient", "age_pred", "pred_error",
            "exp", "exp_error", "confidence"]
    df = pd.merge(dfs[0].loc[:, cols + ["features", "comment"]],
                  dfs[1].loc[:, merge_cols], on="path", suffixes=("_0", "_1"))
    df = pd.merge(df, dfs[2].loc[:, merge_cols], on="path", suffixes=("", "_2"))
    df = df.rename(columns={"exp": "exp_2", "exp_error": "exp_error_2",
                            "confidence": "confidence_2"})

    features = {
        "0": "Amniotic fluid",
        "1": "Apparent size",
        "2": "Appearance of brain",
        "3": "Apparent difficulty in obtaining plane",
        "4": "Ossification of skull",
        "5": "Position within pelvis/uterus",
        "6": "Relationship between the size of the fetus and sector width",
        "7": "Shadow",
        "8": "Shape of skull"
    }
    features_short = {
        "0": "Amniotic fluid",
        "1": "Size",
        "2": "Brain",
        "3": "Plane difficulty",
        "4": "Skull ossification",
        "5": "Pelvis/uterus position",
        "6": "Size vs sector width",
        "7": "Shadow",
        "8": "Skull shape"
    }

    df["feature_names"] = df["features"].apply(
        lambda x: [features[v] for v in x] if x is not None else None)
    df["feature_names_short"] = df["features"].apply(
        lambda x: [features_short[v] for v in x] if x is not None else None)
    df["num_features"] = df.features.apply(lambda x: len(x) if x is not None else 0)

    # weight of advice, trust metric
    def get_woa(initial, final, ai, lower_bound=1):
        out = (initial - final) / (initial - ai)
        if lower_bound is not None:
            out[(initial - ai).abs() <= lower_bound] = np.nan
        return out

    df["woa_12"] = get_woa(df["exp_0"], df["exp_1"], df["age_pred"])
    df["woa_13"] = get_woa(df["exp_0"], df["exp_2"], df["age_pred"])
    df["woa_23"] = get_woa(df["exp_1"], df["exp_2"], df["age_pred"])

    df["woa_12_no_thresh"] = get_woa(df["exp_0"], df["exp_1"], df["age_pred"], lower_bound=None)
    df["woa_13_no_thresh"] = get_woa(df["exp_0"], df["exp_2"], df["age_pred"], lower_bound=None)
    df["woa_23_no_thresh"] = get_woa(df["exp_1"], df["exp_2"], df["age_pred"], lower_bound=None)

    df["pred_abs_error"] = df["pred_error"].abs()
    df["exp_abs_error_0"] = df["exp_error_0"].abs()
    df["exp_abs_error_1"] = df["exp_error_1"].abs()
    df["exp_abs_error_2"] = df["exp_error_2"].abs()

    # df["appropriate_reliance_stage_2"] = df["woa_12"] * np.sign(df["exp_abs_error_1"] - df["pred_abs_error"])
    # df["appropriate_reliance_stage_3"] = df["woa_13"] * np.sign(df["exp_abs_error_2"] - df["pred_abs_error"])
    #
    # def get_appropriate_reliance(initial_error, final_error, model_error, lower_bound=1):
    #     out = (initial_error - final_error) / (initial_error - model_error)
    #     if lower_bound is not None:
    #         out[(initial_error - model_error).abs() <= lower_bound] = np.nan
    #     return out
    #
    # df["appropriate_reliance_v2_stage_2"] = get_appropriate_reliance(df["exp_abs_error_0"], df["exp_abs_error_1"], df["pred_abs_error"])
    # df["appropriate_reliance_v2_stage_3"] = get_appropriate_reliance(df["exp_abs_error_0"], df["exp_abs_error_2"], df["pred_abs_error"])

    df["model_better_0"] = (df["exp_abs_error_0"] > df["pred_abs_error"])
    df["model_better_1"] = (df["exp_abs_error_1"] > df["pred_abs_error"])
    df["model_better_2"] = (df["exp_abs_error_2"] > df["pred_abs_error"])

    df["model_diff_0"] = df["exp_0"] - df["age_pred"]
    df["model_diff_1"] = df["exp_1"] - df["age_pred"]
    df["model_diff_2"] = df["exp_2"] - df["age_pred"]

    df["moves_closer_1"] = df["model_diff_1"].abs() < df["model_diff_0"].abs()
    df["moves_closer_2"] = df["model_diff_2"].abs() < df["model_diff_0"].abs()
    df[" "] = df["model_diff_0"].abs() - df["model_diff_1"].abs()
    df["moves_closer_val_2"] = df["model_diff_0"].abs() - df["model_diff_2"].abs()

    df["over_reliance_1"] = ~df["model_better_0"] & df["moves_closer_1"]
    df["over_reliance_2"] = ~df["model_better_0"] & df["moves_closer_2"]
    df["under_reliance_1"] = df["model_better_0"] & ~df["moves_closer_1"]
    df["under_reliance_2"] = df["model_better_0"] & ~df["moves_closer_2"]
    df["appropriate_reliance_1"] = (df["model_better_0"] & df["moves_closer_1"]) | (~df["model_better_0"] & ~df["moves_closer_1"])
    df["appropriate_reliance_2"] = (df["model_better_0"] & df["moves_closer_2"]) | (~df["model_better_0"] & ~df["moves_closer_2"])

    df["over_reliance_1"].sum(), df["under_reliance_1"].sum(), df["appropriate_reliance_1"].sum()
    df["over_reliance_2"].sum(), df["under_reliance_2"].sum(), df["appropriate_reliance_2"].sum()

    df.loc[df["over_reliance_1"], "moves_closer_val_1"].abs().sum()
    df.loc[df["over_reliance_2"], "moves_closer_val_2"].abs().sum()

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

    df.to_csv(out_dir / "study_results.csv", index=False)

    if args.no_plots:
        print("Ending early!")
        return None

    # TODO: Feature exploration
    #       Compare ROIs

    for suffix in ["", "_no_thresh"]:
        for k in ["woa_12", "woa_13", "woa_23"]:
            key = k + suffix
            sns.displot(df[key])
            plt.savefig(out_dir / (key + "_dist.pdf"))

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for i, ax in enumerate(axes):
        sns.histplot(maes[i], label=f"Exp{i}", ax=ax)
        ax.set_xlabel("MAE / weeks")
    plt.tight_layout()
    plt.savefig(out_dir / "mae_hist.png", bbox_inches="tight")
    plt.show()

    mean_mae = maes.mean(axis=1)
    std_mae = maes.std(axis=1)
    cis_mae = []
    for i in range(4):
        sorted_mae = maes[i].copy()
        sorted_mae.sort()
        ci = (sorted_mae[int(0.05*n_samples)], sorted_mae[int(0.95*n_samples)])
        cis_mae.append(ci)

    mae_df = pd.DataFrame(maes.T, columns=["Stage 1", "Stage 2", "Stage 3", "XAI"])
    mae_df = mae_df.reset_index()
    mae_df = pd.melt(mae_df, id_vars=["index"], var_name="Stage", value_name="MAE")
    mae_df = mae_df.reset_index(drop=True)

    sns.displot(data=mae_df, x="MAE", hue="Stage")
    plt.xlabel("MAE / weeks")
    plt.savefig(out_dir / "mae_displot.png", bbox_inches="tight")
    plt.show()

    sns.displot(data=mae_df.loc[mae_df["Stage"] != "XAI"], x="MAE", hue="Stage")
    plt.xlabel("MAE / weeks")
    plt.savefig(out_dir / "mae_displot_short.png", bbox_inches="tight")
    plt.show()

    for i in range(3):
        print(f"Experiment {i+1} MAE: {mean_mae[i]:.2f} ({cis_mae[i][0]:.2f}-{cis_mae[i][1]:.2f} 95% CI) weeks")
    print()
    for i in range(3):
        print(f"Experiment {i+1} MAE: {mean_mae[i]*7:.2f} ({cis_mae[i][0]*7:.2f}-{cis_mae[i][1]*7:.2f} 95% CI) days")

    ttest0 = ttest_ind(maes[0], maes[1])
    ttest1 = ttest_ind(maes[1], maes[2])
    print()
    print("Experiment 1 vs 2:", ttest0)
    print("Experiment 2 vs 3:", ttest1)


    plt.rcParams.update({'font.size': 14})
    fig, ax = plt.subplots()
    bar = plt.bar(["Image", "Image + AI", "Image + XAI", "XAI"], mean_mae*7, yerr=std_mae*7)
    ax.bar_label(bar, fmt="%.1f")
    plt.ylabel("Mean Average Error / days")
    plt.xlabel("Study Phase")
    ax.set_ylim([0, 27])
    #plt.text(1.25, 10, "p=2e-80", color="red", size=10)
    plt.tight_layout()
    plt.savefig(out_dir / "exp_summary.png", bbox_inches="tight")
    plt.show()

    for i in range(3):
        fig, ax = plt.subplots()
        ax.scatter(range(len(dfs[i])), dfs[i].age, label="GT")
        ax.scatter(range(len(dfs[i])), dfs[i].age_pred, label="model")
        ax.scatter(range(len(dfs[i])), dfs[i][f"exp"], label=f"exp{i+1}")
        plt.legend(frameon=False)
        plt.xlim([-1, 35])
        plt.tight_layout()
        plt.savefig(out_dir / f"exp{i+1}_scatter.png", bbox_inches="tight")
        plt.show()

        fig, ax = plt.subplots()
        ax.scatter(range(len(dfs[i])), dfs[i][f"exp_error"])
        plt.tight_layout()
        plt.savefig(out_dir / f"exp{i+1}_error_scatter.png", bbox_inches="tight")
        plt.show()

        print(dfs[i][f"exp_error"].abs().mean())

        bland_altman(dfs[i]["age_pred"], dfs[i][f"exp"], out_dir / f"bland_altman_exp{i+1}.png")
        #bland_altman(dfs[i]["age"], dfs[i][f"age_pred"])

    for i in range(3):
        fig, ax = plt.subplots()
        ax.scatter(dfs[i]["age"], dfs[i]["age_pred"])
        ax.scatter(dfs[i]["age"], dfs[i]["exp"])
        ax.set_xlabel("Ground Truth GA / Weeks")
        ax.set_ylabel("Predicted GA / Weeks")
        ax.axis("equal")
        plt.tight_layout()
        plt.savefig(out_dir / f"exp{i+1}_association_plot.png", bbox_inches="tight")
        plt.show()

    handles = [
        mlines.Line2D([], [], color=f'C0', marker='o', linewidth=0, markersize=6, label=f'Model'),
        mlines.Line2D([], [], color=f'C1', marker='o', linewidth=0, markersize=6, label=f'Clinician')
    ]

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for i, ax in enumerate(axes):
        ax.scatter(dfs[i]["age"], dfs[i]["age_pred"])
        ax.scatter(dfs[i]["age"], dfs[i]["exp"])
        ax.axis("equal")
        ax.set_xlabel("Ground Truth Age / Weeks")
    axes[0].set_ylabel("Predicted Age / weeks")
    plt.legend(handles=handles, frameon=False)
    plt.tight_layout()
    plt.savefig(out_dir / "association_plot.png", bbox_inches="tight")
    plt.show()

    max_v = max([v["exp_error"].abs().max() for v in dfs]) + 0.2
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for i, ax in enumerate(axes):
        ax.axhline(0, color="black", linestyle="--", alpha=0.6)
        ax.scatter(dfs[i]["age"], dfs[i]["pred_error"])
        ax.scatter(dfs[i]["age"], dfs[i]["exp_error"])
        ax.set_xlabel("Ground Truth Age / Weeks")
        ax.set_xlim([13, 42])
        ax.set_ylim([-max_v, max_v])
    axes[0].set_ylabel("Residual / weeks")
    plt.legend(handles=handles, frameon=False)
    plt.tight_layout()
    plt.savefig(out_dir / "residual_plot.png", bbox_inches="tight")
    plt.show()

    # fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    # for i, ax in enumerate(axes):
    #     i = i + 1
    #     ax.axhline(0, color="black", linestyle="--", alpha=0.6)
    #     ax.scatter(dfs[i]["age"], dfs[i]["pred_error"])
    #     ax.scatter(dfs[i]["age"], dfs[i]["exp_error"])
    #     ax.set_xlabel("Ground Truth Age / Weeks")
    #     ax.set_xlim([13, 42])
    #     ax.set_ylim([-7, 7])
    # axes[0].set_ylabel("Residual / weeks")
    # plt.legend(handles=handles, frameon=False)
    # plt.tight_layout()
    # plt.show()

    slope, intercept, r_value, p_value, std_err = linregress(dfs[1]["exp"], dfs[1]["age"])

    for i in range(3):
        #print(data[i]["seconds"][1:].mean())
        print(data[i]["seconds"][2:55].mean())

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for i, ax in enumerate(axes):
        ax.plot(data[i]["seconds"][1:])
        ax.plot(range(1, 55-1), data[i]["seconds"][2:55])
        ax.set_xlabel("Image")
        ax.set_ylabel("Time / s")
    plt.tight_layout()
    plt.savefig(out_dir / "time_plot.png", bbox_inches="tight")
    plt.show()

    # bins0 = np.arange(-14, 16, 2)
    # bins1 = np.arange(-7, 8)
    # fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    # for i, ax in enumerate(axes):
    #     if i == 0:
    #         bins = bins0
    #     else:
    #         bins = bins1
    #     sns.histplot(dfs[i]["exp_error"], ax=ax, bins=bins)
    #     ax.set_xlabel("Residual / weeks")
    # plt.tight_layout()
    # plt.show()



    # img_dir = Path("/home/lina3782/data00/intergrowth/original")
    # img_out_dir = out_dir / "labelled_images"
    # img_out_dir.mkdir(exist_ok=True)

    """
    for i in range(len(df)):
        row = df.loc[i]
        img = cv2.imread(str(img_dir / row["path"]))
        fig, ax = plt.subplots()
        ax.imshow(img)
        plt.title(f"GT: {row['age']:.1f}, AI: {int(row['age_pred']): d}, Exp: {row['exp_0']}, {row['exp_1']}, {row['exp_2']}")
        ax.axis("off")
        plt.tight_layout()
        plt.savefig(img_out_dir / row["path"])
    """
    fig, ax = plt.subplots()
    markers = ["d", "x", "+"]
    for i in range(3):
        ax.scatter(df.index, df[f"exp_{i}"], marker=markers[i], label=f"Exp{i}")
    ax.scatter(df.index, df["age"], color="red", marker="1", label="GT")
    plt.xlabel("Image")
    plt.ylabel("Prediction / weeks")
    plt.xlim(-1, 65)
    plt.legend()
    plt.tight_layout()
    plt.show()

    fig, ax = plt.subplots()
    markers = ["d", "x", "+"]
    plt.axhline(0, color="black", linestyle="--")
    for i in range(3):
        ax.scatter(df.index, df[f"exp_error_{i}"], marker=markers[i], label=f"Exp{i}")
    plt.xlabel("Image")
    plt.ylabel("Residual / weeks")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "changes_residual_plot.png", bbox_inches="tight")
    plt.show()

    # sub_df = df.loc[df["exp_1"] != df["exp_2"]]
    # fig, ax = plt.subplots()
    # markers = ["d", "x", "+"]
    # colors = ["C0", "C1", "C2"]
    # plt.axhline(0, color="black", linestyle="--")
    # for i in range(3):
    #     if i != 0:
    #         ax.scatter(range(len(sub_df)), sub_df[f"exp_error_{i}"], marker=markers[i], color=colors[i], label=f"Exp{i}")
    # plt.xlabel("Image")
    # plt.ylabel("Residual / weeks")
    # plt.legend()
    # plt.tight_layout()
    # plt.show()

    confidence_dfs = [dfs[i][["confidence",]] for i in range(3)]
    for i in range(3):
        confidence_dfs[i]["Stage"] = f"Stage {i+1}"
    confidence_df = pd.concat(confidence_dfs).reset_index(drop=True)
    confidence_df["confidence"] = confidence_df["confidence"] + 1

    bins = np.array([0.6, 1.4, 1.6, 2.4, 2.6, 3.4, 3.6, 4.4, 4.6, 5.5])

    sns.displot(data=confidence_df, x="confidence", hue="Stage", multiple="dodge", bins=bins, hue_order=[f"Stage {i+1}" for i in range(3)])
    plt.savefig(out_dir / "confidence_barplot.png", bbox_inches="tight")
    plt.show()

    from copy import deepcopy

    colors = sns.color_palette("seismic", n_colors=5) + ["grey"]

    confidence_df = deepcopy(confidence_df)
    confidence_df.loc[confidence_df["confidence"].isnull(), "confidence"] = 10
    confidence_df["confidence"] = confidence_df["confidence"].astype(int).astype(str)
    confidence_df.loc[confidence_df["confidence"] == "10", "confidence"] = "Missing"
    sns.displot(data=confidence_df, x="Stage", hue="confidence", multiple="stack", hue_order=["1", "2", "3", "4", "5", "Missing"], palette=colors)
    plt.savefig(out_dir / "confidence_proportion_fillplot.png", bbox_inches="tight")
    plt.show()

    all_features = df["feature_names_short"].values
    all_features_expanded = []
    for v in all_features:
        if v is not None:
            all_features_expanded += v

    all_features_series = pd.Categorical(all_features_expanded, categories=features_short.values())

    sns.countplot(all_features_series, color="C0")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(out_dir / "features_hist.png", bbox_inches="tight")
    plt.show()

    sns.displot(df["num_features"], bins=np.arange(0, 10) - 0.5)
    plt.xlabel("No. Features")
    plt.tight_layout()
    plt.savefig(out_dir / "num_features_hist.png", bbox_inches="tight")
    plt.show()

    #
    # img_dir = Path("/home/lina3782/labs/protopnet/intergrowth/saved_models/resnet18/017/195_19push0.6905.pth_test_results/reader_study/via_speed_test_2/utility_plain")
    # orig_img_dir = Path("/home/lina3782/data00/intergrowth/original")
    #
    # for i in range(4, 10):
    #     plot_bboxes_on_orig_image(i, img_dir, orig_img_dir, df, data)
    #
    # shifted_bboxes = data[0]["xy"]
    # bboxes = []
    # for i in range(len(shifted_bboxes)):
    #     if len(shifted_bboxes[i]) == 0:
    #         bboxes.append([])
    #     else:
    #         img_bboxes = []
    #         img = cv2.imread(str(img_dir / f"{i:04d}.png"))
    #         orig_img = cv2.imread(str(orig_img_dir / df.loc[i, "path"]))
    #         for bbox_info in shifted_bboxes[i]:
    #             bbox_info = [int(v) for v in bbox_info[1:]]
    #             new_bbox = convert_coords_to_orig_image(img, orig_img, bbox_info)
    #             img_bboxes.append(new_bbox)
    #         bboxes.append(img_bboxes)
    #
    #
    # df_subset = df.loc[df.exp_2 != df.exp_1].copy()
    # df_subset["exp12_diff"] = df_subset["exp_2"] - df_subset["exp_1"]
    # df_subset["exp12_error_diff"] = df_subset["exp_error_2"] - df_subset["exp_error_1"]
    # df_subset["exp12_ai_diff"] = (df_subset["exp_2"] - df_subset["age_pred"]).abs() - (df_subset["exp_1"] - df_subset["age_pred"]).abs()
    # disagree_improvement = df_subset.loc[(df_subset["exp12_ai_diff"] > 0) & (df_subset["exp12_error_diff"] < 0), "exp12_error_diff"].abs().mean()
    # agree_improvement = df_subset.loc[
    #     (df_subset["exp12_ai_diff"] < 0) & (df_subset["exp12_error_diff"] < 0), "exp12_error_diff"].abs().mean()
    # disagree_avg = df_subset.loc[(df_subset["exp12_ai_diff"] > 0), "exp12_error_diff"].mean()
    # agree_avg = df_subset.loc[(df_subset["exp12_ai_diff"] < 0), "exp12_error_diff"].mean()
    #

    print("Done!")


def plot_bboxes_on_orig_image(i, img_dir, orig_img_dir, df, data):
    img = cv2.imread(str(img_dir / f"{i:04d}.png"))
    orig_img = cv2.imread(str(orig_img_dir / df.loc[i, "path"]))
    fig, ax = plt.subplots()
    for bbox_info in data[0]["xy"][i]:
        bbox_info = [int(v) for v in bbox_info[1:]]
        new_bbox = convert_coords_to_orig_image(img, orig_img, bbox_info)
        orig_img = add_bbox(orig_img, new_bbox)
    ax.imshow(orig_img)
    plt.title(
        f"GT: {df.loc[i, 'age']:.1f}, "
        f"AI: {int(df.loc[i, 'age_pred']): d}, "
        f"Exp: {df.loc[i, 'exp_0']}, "
        f"{df.loc[i, 'exp_1']}, "
        f"{df.loc[i, 'exp_2']}")
    ax.axis("off")
    plt.tight_layout()
    plt.show()


def convert_coords_to_orig_image(img, orig_img, bbox_info):
    white_img = img == 255
    white_img = white_img.all(axis=2)
    coords = np.where(~white_img)
    l, r = coords[1].min(), coords[1].max()
    t, b = coords[0].min(), coords[0].max()

    width = r - l
    height = b - t
    width_ratio = orig_img.shape[1]/width
    height_ratio = orig_img.shape[0] / height

    x, y, w, h = bbox_info

    w1 = int(w * width_ratio)
    h1 = int(h * height_ratio)
    x1 = int((x-l)*width_ratio)
    y1 = int((y-t)*height_ratio)
    return [x1, y1, w1, h1]


def add_bbox(img, bbox_info, color=(255, 255, 0)):
    x, y, w, h = bbox_info
    #img_bgr_uint8 = cv2.cvtColor(np.uint8(img), cv2.COLOR_RGB2BGR)
    cv2.rectangle(img, (x, y), (x+w, y+h), color, thickness=2)
    return img


def bland_altman(x1, x2, savefig=None):
    avg = (x1 + x2)/2
    diff = x2 - x1
    sd = diff.std()

    fig, ax = plt.subplots()
    ax.scatter(avg, diff)
    plt.axhline(0, color="black", linestyle="--")
    plt.axhline(1.96 * sd, color="red", linestyle="--")
    plt.axhline(-1.96 * sd, color="red", linestyle="--")
    plt.xlabel("Average / Weeks")
    plt.ylabel("Difference (Exp - Model) / Weeks")
    if savefig is not None:
        plt.savefig(savefig, bbox_inches="tight")
    plt.show()


def plot_summary(seconds, s=10, e=25):
    print(seconds.mean())
    print(seconds[s:e].mean())

    fig, ax = plt.subplots()
    ax.plot(seconds)
    ax.plot(range(s, e), seconds[s:e])
    ax.set_xlabel("Image")
    ax.set_ylabel("Time / s")
    plt.show()

    sns.displot(seconds)
    plt.show()

    sns.displot(seconds[s:e])
    plt.show()


def load_and_extract_jsons(json_path, first_exp=False):
    with open(json_path, "r") as fp:
        results = json.load(fp)
    metadata = [v for v in results["metadata"].values() if v["xy"] == []]
    metadata = [[v for v in metadata if v["vid"] == i][0] for i in results["project"]["vid_list"]]
    ages = [int(v["av"]["1"]) for v in metadata]
    confidence = [int(v["av"]["2"]) if "2" in v["av"].keys() else None for v in metadata]

    times = [[v["time"].split(".")[0] for v in [in_v for in_v in results["metadata"].values() if in_v["vid"] == i] if
              "time" in v.keys()] for i in results["project"]["vid_list"]]
    times = [v[0] if len(v) > 0 else None for v in times]
    times = [datetime.strptime(v, "%Y-%m-%dT%H:%M:%S") if v is not None else None for v in times]
    # If the time is missing, replace it with the average of either side
    # Assumes you don't have two missing in a row
    for i in range(len(times)):
        if times[i] is None:
            if i == 0:
                times[i] = times[i+1]
            elif i == len(times) - 1:
                times[i] = times[i-1]
            else:
                times[i] = times[i-1] + (times[i-1] - times[i+1]) / 2
    d_t = [times[i] - times[i - 1] for i in range(1, len(times))]
    seconds = np.array([v.seconds for v in d_t])
    output = {"ages": ages, "confidence": confidence, "times": times, "seconds": seconds}

    if first_exp:
        features = [v["av"]["3"] if "3" in v["av"].keys() else None for v in metadata]
        comment = [v["av"]["4"] if "4" in v["av"].keys() else None for v in metadata]
        output["features"] = features
        output["comment"] = comment

        data = [v for v in results["metadata"].values() if len(v["xy"]) > 0]
        ordered_data = []
        for i in results["project"]["vid_list"]:
            vid_data = [v["xy"] for v in data if v["vid"] == i]
            ordered_data.append(vid_data)
        output["xy"] = ordered_data

    return output


def bootstrap_mae(pred, gt, rng=np.random.default_rng(), n_samples=100, sample_size=0.6):
    pred = np.array(pred)
    gt = np.array(gt)
    dataset_size = len(pred)
    n_per_sample = int(sample_size * dataset_size)
    indicies = list(range(dataset_size))
    chosen_indicies = np.zeros((n_samples, n_per_sample), dtype=int)
    for i in range(n_samples):
        chosen_indicies[i] = rng.choice(indicies, replace=False, size=n_per_sample)
    chosen_pred = pred[chosen_indicies]
    chosen_gt = gt[chosen_indicies]
    mae = np.abs((chosen_pred - chosen_gt)).mean(axis=1)
    return mae


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--study-id", help="study_id for the participant to be analysed")
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="End the script early so don't update the plots and just output the .csv"
    )
    args = parser.parse_args()
    main(args)
