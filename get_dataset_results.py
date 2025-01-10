
from argparse import ArgumentParser
from pathlib import Path
import os
import re
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from settings import train_push_dir, test_dir
from collate_global_explanation import get_weight_df
from preprocess import mean, std


def main(args):
    if args.dataset == "train":
        path = Path(train_push_dir)
        dataset_name = "train"
    elif args.dataset == "test":
        path = Path(test_dir)
        dataset_name = "test"
    else:
        path = Path(args.dataset) # /home/lina3782/labs/protopnet/interbio/datasets/all
        dataset_name = args.dataset_name
        if dataset_name is None:
            raise ValueError("Must provide --dataset-name if --dataset is a path.")

    if not path.exists():
        raise FileExistsError(f"Path {path} does not exist. Ensure dataset is present at location.")
    else:
        print(f"Loading dataset from {path}")

    if args.class_names is not None:
        with open(args.class_names, "r") as fp:
            class_names = fp.read().split("\n")
    else:
        class_names = None

    load_model_path = Path(args.model)
    load_model_dir = load_model_path.parent
    load_model_name = load_model_path.name

    epoch_number_str = re.search(r'\d+', load_model_name).group(0)
    start_epoch_number = int(epoch_number_str)

    out_dir = load_model_dir / f"{load_model_name}_{dataset_name}_results"
    out_dir.mkdir(exist_ok=True)

    # load the model
    print('load model from ' + str(load_model_path))
    ppnet = torch.load(load_model_path)
    ppnet = ppnet.cuda()
    ppnet.eval()

    weights = ppnet.last_layer.weight.detach().cpu().numpy()
    dataloader = create_dataloader(str(path), ppnet.img_size, args.batch_size, False, args.workers)
    dataset_length = len(dataloader.dataset)

    img_paths, targets = zip(*dataloader.dataset.imgs)

    if not args.load_data:
        img_results = [{"path": Path(path).name, "target": target} for path, target in dataloader.dataset.imgs]
        similarities = np.zeros((dataset_length, ppnet.num_prototypes))
        similarity_maps = np.zeros((dataset_length, ppnet.num_prototypes, 7, 7))
        logits = np.zeros((dataset_length, ppnet.num_classes))

        i = 0
        for batch_idx, (image, label) in enumerate(dataloader):
            with torch.no_grad():
                output, min_distances, sim_maps = ppnet(image.cuda())

            next_i = i + len(image)
            logits[i:next_i] = output.cpu().numpy()
            similarities[i:next_i] = ppnet.distance_2_similarity(min_distances).cpu().numpy()
            similarity_maps[i:next_i] = sim_maps.detach().cpu().numpy()
            i += len(image)

        contributions = similarities[:, :, np.newaxis] * weights.T  # N, P, C
        logits2 = contributions.sum(axis=1)
        assert np.allclose(logits, logits2, atol=3e-2)
        predictions = logits.argmax(axis=1)
        correct = (predictions == targets)
        acc = correct.sum() / len(correct)
        print(f"Accuracy: {acc:.3f}")

        for i in range(len(img_results)):
            img_results[i]["prediction"] = int(predictions[i])
            img_results[i]["correct"] = bool(correct[i])
            img_results[i]["similarities"] = similarities[i].tolist()
            # These are a little too big...
            # img_results[i]["similarity_maps"] = similarity_maps[i].tolist()
            # img_results[i]["contributions"] = contributions[i].tolist()
            img_results[i]["logits"] = logits[i].tolist()

        out_results = {
            "dataset_name": dataset_name,
            "dataset_path": str(path),
            "model_path": str(load_model_path),
            "acc": acc,
            "img_results": img_results
        }
        # Save the bigger arrays as .npy, rather than in the .json
        if not args.no_npy:
            np.save(str(out_dir / "similarities.npy"), similarities)  # This is actually in both
            np.save(str(out_dir / "contributions.npy"), contributions)
            np.save(str(out_dir / "similarity_maps.npy"), similarity_maps)
        del similarity_maps

        with open(out_dir / "results.json", "w") as fp:
            json.dump(out_results, fp, indent=2)
    else:
        contributions = np.load(str(out_dir / "contributions.npy"))
        similarities = np.load(str(out_dir / "similarities.npy"))
        logits2 = contributions.sum(axis=1)
        predictions = logits2.argmax(axis=1)

    weight_df = get_weight_df(ppnet, class_names)
    non_zero_weight_df = weight_df.loc[weight_df.weight.abs() > 0]
    class_prototypes = non_zero_weight_df.groupby("class").prototype.unique()
    prototype_classes = {}
    for prototype in range(ppnet.num_prototypes):
        ps = []
        for class_id in range(ppnet.num_classes):
            if prototype in class_prototypes[class_id]:
                ps.append(class_id)
        prototype_classes[prototype] = ps

    prototype = 1
    class_id = 0
    df = pd.DataFrame(contributions[:, prototype, class_id])  # N, P, C
    df["class"] = targets
    df["class"] = df["class"].astype(str)
    df["same_class"] = df["class"] == class_id
    sns.displot(df, x=0, hue="class", kind="kde")
    plt.show()

    prototype = 13
    df = pd.DataFrame(similarities[:, prototype])  # N, P
    df["class"] = targets
    df["class"] = df["class"].astype(str)
    df["non_zero_class"] = df["class"].astype(int).isin(prototype_classes[prototype])

    sns.displot(df, x=0, hue="non_zero_class", kind="kde")
    plt.ylim([0, 0.25])
    plt.xticks(range(10))
    plt.xlabel("Similarity")
    plt.show()

    sns.displot(df.loc[df.non_zero_class], x=0, hue="class", kind="kde")
    #plt.ylim([0, 0.25])
    plt.xticks(range(10))
    plt.xlabel("Similarity")
    plt.show()

    mean_sims = np.zeros((similarities.shape[1], contributions.shape[2]))
    #correct_mean_sims = np.zeros((similarities.shape[1], contributions.shape[2]))
    #incorrect_mean_sims = np.zeros((similarities.shape[1], contributions.shape[2]))
    bins = np.linspace(min(0, similarities.min()), int(similarities.max()) + 1, 100)
    contribution_bins = np.linspace(contributions.min(), int(contributions.max()) + 1, 100)

    for prototype in range(ppnet.num_prototypes):
        df = pd.DataFrame(similarities[:, prototype])  # N, P
        df["class"] = targets
        df["class"] = df["class"].astype(str)
        df["target"] = targets
        df["pred"] = predictions
        df["prototype"] = prototype
        df["non_zero_class"] = df["class"].astype(int).isin(prototype_classes[prototype])

        mean_sims[prototype] = get_mean_sim(df)
        #correct_mean_sims[prototype] = get_mean_sim(df.loc[df["target"] == df["pred"]])
        #incorrect_mean_sims[prototype] = get_mean_sim(df.loc[df["target"] != df["pred"]])

        if args.plot_proto:
            sub_dir = out_dir / f"{prototype:02d}"
            sub_dir.mkdir(exist_ok=True, parents=True)

            sns.displot(df, x=0, kind="kde", hue="non_zero_class")
            plt.ylim([0, 0.5])
            plt.xticks(range(10))
            plt.xlabel("Similarity")
            plt.savefig(sub_dir / f"similarity_non_zero_kde.png")

            # plot_similarities(df, ppnet.num_classes, sub_dir, bins)
            plot_contributions(contributions[:, prototype, :],  ppnet.num_classes, sub_dir, contribution_bins)

            plot_mean_sim_across_class(mean_sims[prototype], sub_dir, "mean_similarity_across_class.png")
            """
            plot_mean_sim_across_class(
                correct_mean_sims[prototype],
                sub_dir, "mean_similarity_across_class_correct.png"
            )

            plot_mean_sim_across_class(
                incorrect_mean_sims[prototype],
                sub_dir, "mean_similarity_across_class_incorrect.png"
            )
            """
            sns.displot(contributions[:, prototype, :], kind="kde", warn_singular=False)
            plt.ylim([0, 0.5])
            #plt.xticks(range(10))
            plt.xlabel("Contribution")
            plt.savefig(sub_dir / f"contributions_kde.png")

            plt.close("all")

    fig, ax = plt.subplots()
    for prototype in range(ppnet.num_prototypes):
        ax.plot(mean_sims[prototype])
    plt.savefig(out_dir / "mean_similarities_across_class.png")

    fig, ax = plt.subplots()
    for prototype in range(ppnet.num_prototypes):
        ax.plot(contributions[:, prototype].mean(axis=0))
    plt.savefig(out_dir / "mean_contributions_across_logit.png")

    contributions_df = create_contributions_df(contributions, ppnet.num_classes)
    contributions_pred_logit = contributions.swapaxes(2, 1)[np.eye(ppnet.num_classes)[predictions] == 1]
    contributions_pred_logit.sort(axis=1)
    top_n_contribution_means = np.zeros(10)
    for n in range(10):
        top_n_contribution_means[n] = contributions_pred_logit[:, -(n+1):].sum(axis=1).mean()

    top_n_contribution_means_normalised = [0] + (top_n_contribution_means / contributions_pred_logit.sum(axis=1).mean()).tolist()
    np.save(str(out_dir / "top_n_contribution_means_normalised.npy"), np.array(top_n_contribution_means_normalised))

    fig, ax = plt.subplots()
    plt.plot(top_n_contribution_means_normalised)
    plt.grid()
    plt.xlabel("No. prototypes")
    plt.ylabel("Explanation completeness")
    plt.xticks(range(11))
    plt.tight_layout()
    plt.savefig(out_dir / "total_normalised_contribution_across_top-n.png")
    plt.close(fig)

    data_20 = np.load(str(out_dir.parent.parent /
                          "pruned_prototypes_epoch195_k6_ct0_wt0.2000_000/"
                          "195_19push0.6905_15_prune0.6818.pth_test_results/"
                          "top_n_contribution_means_normalised.npy"))
    data_25 = np.load(str(out_dir.parent.parent /
                      "pruned_prototypes_epoch195_k6_ct0_wt0.2500_000/"
                      "195_19push0.6905_15_prune0.6589.pth_test_results/"
                      "top_n_contribution_means_normalised.npy"))
    data_orig = np.load(str(out_dir.parent.parent /
                      "195_19push0.6905.pth_test_results/"
                      "top_n_contribution_means_normalised.npy"))

    # MAE on the 65 images from prelim AGE study
    # none: 6.699999999999945
    # 020: 7.115384615384553
    # 025: 7.507692307692254

    # MAE on INTERGROWTH-21st
    # none: 6.27
    # 020: 6.33
    # 025: 6.84

    plt.rcParams.update({'font.size': 14})
    fig, ax = plt.subplots()
    plt.axvline(4, 0, 1, linestyle="--", color="gray")
    plt.plot(data_orig, label="None")
    plt.plot(data_20, label="0.20")
    plt.plot(data_25, label="0.25")
    plt.grid()
    plt.xlabel("No. prototypes")
    plt.ylabel("Explanation completeness")
    plt.xticks(range(11))
    plt.legend(title="Pruning")
    plt.tight_layout()
    plt.ylim([0, 1.01])
    plt.xlim([0, 10])
    plt.text(8, 0.6, "6.3 days", color="C0")
    plt.text(7.7, 0.82, "6.3 days", color="C1")
    plt.text(5.5, 0.85, "6.8 days", color="C2")
    plt.savefig(out_dir / "total_normalised_contribution_across_top-n_comparison_intergrowth.png", bbox_inches="tight")
    plt.show()
    plt.close(fig)

    #correct_con_df = create_contributions_df(contributions[correct])
    #incorrect_con_df = create_contributions_df(contributions[~correct])

    plot_stacked_contributions(
        contributions_df,
        class_names,
        out_dir,
        ppnet,
        "mean_contributions_across_logit_stacked_bar_fig.png",
        True,
        colours="Spectral",
        neg_colours="crest_r"
    )
    plt.show()

    plot_stacked_contributions(
        contributions_df,
        class_names,
        out_dir,
        ppnet,
        "mean_contributions_across_logit_stacked_bar_same_color_fig.png",
        False,
        colours="Spectral"
    )
    plt.show()

    plot_stacked_sims(
        mean_sims,
        class_names,
        out_dir,
        ppnet,
        "mean_sim_stacked_bar.png",
        "viridis",
    )



    """
    plot_stacked_contributions(
        correct_con_df,
        class_names,
        out_dir,
        ppnet,
        "mean_contributions_across_logit_stacked_bar_correct.png"
    )

    plot_stacked_contributions(
        incorrect_con_df,
        class_names,
        out_dir,
        ppnet,
        "mean_contributions_across_logit_stacked_bar_incorrect.png"
    )


    plot_stacked_sims(
        correct_mean_sims,
        class_names,
        out_dir,
        ppnet,
        "mean_sim_stacked_bar_correct.png"
    )

    plot_stacked_sims(
        incorrect_mean_sims,
        class_names,
        out_dir,
        ppnet,
        "mean_sim_stacked_bar_incorrect.png"
    )
    """
    fig, ax = plt.subplots()
    p_colors = sns.color_palette("viridis", n_colors=ppnet.num_prototypes)
    n_colors = sns.color_palette("flare_r", n_colors=ppnet.num_prototypes)
    p_bottom = np.zeros(ppnet.num_classes)
    n_bottom = np.zeros(ppnet.num_classes)
    for p in range(ppnet.num_prototypes):
        data = weights[:, p]
        p_data = [v if v > 0 else 0 for v in data]
        n_data = [v if v < 0 else 0 for v in data]
        ax.bar(class_names, p_data, bottom=p_bottom, color=p_colors[p], label=p)
        ax.bar(class_names, n_data, bottom=n_bottom, color=n_colors[p], label=p)
        p_bottom += p_data
        n_bottom += n_data
    plt.xlabel("Class")
    plt.ylabel("Total weight")
    plt.xticks(rotation=45, horizontalalignment='right', rotation_mode="anchor")
    #plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(out_dir / "weights_stacked_bar.png")

    print("Done!")


def get_mean_sim(df):
    return df.groupby("target")[0].mean().values


def plot_mean_sim_across_class(mean_sim, sub_dir, name):
    fig, ax = plt.subplots()
    ax.plot(mean_sim)
    plt.ylim([0, mean_sim.max() * 1.2])
    plt.xlabel("Class")
    plt.ylabel("Mean similarity")
    plt.savefig(sub_dir / name)
    del fig
    del ax


def plot_stacked_sims(mean_sims, class_names, out_dir, ppnet, name, color_palette="viridis"):
    fig, ax = plt.subplots()
    colors = sns.color_palette(color_palette, n_colors=ppnet.num_prototypes)
    bottom = np.zeros(ppnet.num_classes)
    for p in range(ppnet.num_prototypes):
        data = mean_sims[p, :]
        ax.bar(class_names, data, bottom=bottom, color=colors[p], label=p)
        bottom += data
    plt.xlabel("Class")
    plt.ylabel("Total mean similarity")
    plt.xticks(rotation=45, horizontalalignment='right', rotation_mode="anchor")
    plt.tight_layout()
    plt.savefig(out_dir / name)


def ugly_plot_stacked_sims(mean_sims, class_names, out_dir, ppnet, name, color_palette="viridis", shift=2):
    fig, ax = plt.subplots()
    color_names = ["Blues", "Reds", "Greens"]
    n = len(color_names)
    palettes = [sns.color_palette(palette, n_colors=ppnet.num_prototypes//n + 1 + shift) for palette in color_names]
    palettes = [palette[shift:] for palette in palettes]
    for palette in palettes:
        np.random.shuffle(palette)
    colors = [palettes[i % n][i // n] for i in range(ppnet.num_prototypes)]
    bottom = np.zeros(ppnet.num_classes)
    for p in range(ppnet.num_prototypes):
        color = colors[p]
        data = mean_sims[p, :]
        ax.bar(class_names, data, bottom=bottom, color=color, label=p)
        bottom += data
    plt.xlabel("Class")
    plt.ylabel("Total mean similarity")
    plt.xticks(rotation=45, horizontalalignment='right', rotation_mode="anchor")
    plt.tight_layout()
    plt.savefig(out_dir / name)


def ugly_plot_stacked_contributions(contributions_df, class_names, out_dir, ppnet, name, shift=4):
    fig, ax = plt.subplots()
    color_names = ["viridis"]
    n = len(color_names)
    palettes = [sns.color_palette(palette, n_colors=ppnet.num_prototypes//n + 1 + shift) for palette in color_names]
    palettes = [palette[shift:] for palette in palettes]
    for palette in palettes:
        np.random.shuffle(palette)
    colors = [palettes[i % n][i // n] for i in range(ppnet.num_prototypes)]
    p_bottom = np.zeros(ppnet.num_classes)
    n_bottom = np.zeros(ppnet.num_classes)
    for p in range(ppnet.num_prototypes):
        data = contributions_df.loc[contributions_df.prototype == p, "contribution"].values
        p_data = [v if v > 0 else 0 for v in data]
        n_data = [v if v < 0 else 0 for v in data]
        ax.bar(class_names, p_data, bottom=p_bottom, color=colors[p], label=p)
        ax.bar(class_names, n_data, bottom=n_bottom, color=colors[p], label=p)
        p_bottom += p_data
        n_bottom += n_data
    plt.xlabel("Class")
    plt.ylabel("Total mean contribution")
    plt.xticks(rotation=45, horizontalalignment='right', rotation_mode="anchor")
    plt.tight_layout()
    plt.savefig(out_dir / name)


def plot_stacked_contributions(contributions_df, class_names, out_dir, ppnet, name, diff_neg_color=True, colours="viridis", neg_colours="flare_r"):
    fig, ax = plt.subplots()
    p_colors = sns.color_palette(colours, n_colors=ppnet.num_prototypes)
    n_colors = sns.color_palette(neg_colours, n_colors=ppnet.num_prototypes)
    p_bottom = np.zeros(ppnet.num_classes)
    n_bottom = np.zeros(ppnet.num_classes)
    for p in range(ppnet.num_prototypes):
        data = contributions_df.loc[contributions_df.prototype == p, "contribution"].values
        p_data = [v if v > 0 else 0 for v in data]
        n_data = [v if v < 0 else 0 for v in data]
        ax.bar(class_names, p_data, bottom=p_bottom, color=p_colors[p], label=p)
        if diff_neg_color:
            n_color = n_colors[p]
        else:
            n_color = p_colors[p]
        ax.bar(class_names, n_data, bottom=n_bottom, color=n_color, label=p)
        p_bottom += p_data
        n_bottom += n_data
    y_min, y_max = ax.get_ylim()
    ax.set_ylim(y_min - 0.1, y_max + 0.1)
    plt.xlabel("Class")
    plt.ylabel("Total mean contribution")
    plt.xticks(rotation=45, horizontalalignment='right', rotation_mode="anchor")
    #plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(out_dir / name)


def create_contributions_df(contributions, n_classes):
    mean_contributions = contributions.mean(axis=0)
    contributions_df = pd.DataFrame(mean_contributions, columns=[f"contribution{i}" for i in range(n_classes)])
    contributions_df["prototype"] = contributions_df.index
    contributions_df = pd.wide_to_long(contributions_df, "contribution", "prototype", "class").reset_index()
    return contributions_df


def plot_contributions(contributions, num_classes, out_dir, bins):
    for class_id in range(num_classes):
        data = contributions[:, class_id]
        if (data == 0).all():
            continue
        else:
            sns.displot(data, kind="hist", bins=bins)
            plt.xticks(range(int(bins.min()), int(bins.max()) + 1))
            plt.xlabel("Contribution")
            plt.savefig(out_dir / f"contributions_class{class_id:02d}.png")

            lower_than = np.where(bins < contributions.min())[0]
            if len(lower_than) > 0:
                min_bin = lower_than[-1]
            else:
                min_bin = 0
            higher_than = np.where(bins > contributions.max())[0]
            if len(higher_than) > 0:
                max_bin = higher_than[0]
            else:
                max_bin = -1
            sns.displot(data, kind="hist", bins=bins[min_bin:max_bin])
            plt.xlabel("Contribution")
            plt.savefig(out_dir / f"contributions_not_fixed_class{class_id:02d}.png")


def plot_similarities(df, num_classes, out_dir, bins):
    for class_id in range(num_classes):
        sns.displot(df.loc[df["class"] == str(class_id)], x=0, kind="hist", bins=bins)
        plt.xticks(range(int(bins.max()) + 1))
        plt.xlabel("Similarity")
        plt.savefig(out_dir / f"similarity_class{class_id:02d}.png")


def create_dataloader(path: str, img_size: int = 224, batch_size: int = 64, shuffle: bool = True, workers: int = 4,
                      normalise: bool = True):
    transforms_list = [
            transforms.Resize(size=(img_size, img_size)),
            transforms.ToTensor(),
        ]
    if normalise:
        transforms_list.append(transforms.Normalize(mean=mean, std=std))
    dataset = datasets.ImageFolder(
        path,
        transforms.Compose(transforms_list))
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle,
        num_workers=workers, pin_memory=False)
    return loader


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--gpuid', help="Which GPU to use", nargs=1, type=str, default=None)
    parser.add_argument('--model', help="Path to the model (.pth)")
    parser.add_argument("--batch-size", help="Batch size for dataloaders", default=256, type=int)
    parser.add_argument("--workers", help="No. workers for dataloading", default=4, type=int)
    parser.add_argument("--dataset", help="Dataset str or path", default="train")
    parser.add_argument("--class-names", default=None, help="Path to .txt containing \\n separated class names")
    parser.add_argument("--dataset-name", default=None, help="Provide if giving a path for --dataset")
    parser.add_argument("--plot-proto", action="store_true", help="Plot distributions for individual prototypes")
    parser.add_argument("--load-data", action="store_true", help="Don't recalculate the data")
    parser.add_argument("--no-npy", action="store_true", help="Don't save the .npy data")
    args = parser.parse_args()
    if args.gpuid is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpuid[0]
    main(args)
