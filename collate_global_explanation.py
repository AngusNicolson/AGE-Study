
from pathlib import Path
from argparse import ArgumentParser
import re

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from PIL import Image
import torch


def main(args):
    n = args.n
    # Create paths and check required files/directories exist
    model_path = Path(args.model)
    if not model_path.exists():
        raise FileExistsError(f"Model {model_path} does not exist.")
    ppnet = torch.load(args.model)
    model_dir = model_path.parent
    model_name = model_path.stem
    epoch = model_name.split("_")[0]
    img_dir = model_dir / "img" / f"epoch-{epoch}"
    if not img_dir.exists():
        raise FileExistsError(f"Directory containing the prototypes does not exist: {img_dir}")
    train_dir = model_dir / f"{model_name}_nearest_train"
    test_dir = model_dir / f"{model_name}_nearest_test"
    if not (test_dir.exists() and train_dir.exists()):
        raise FileExistsError(f"Global analysis output not in {model_dir}")

    if args.class_names is not None:
        with open(args.class_names, "r") as fp:
            class_names = fp.read().split("\n")
    else:
        class_names = None
    out_dir = model_dir / f"{model_name}_global"
    out_dir.mkdir(exist_ok=True)
    if args.indiv:
        indiv_out_dir = out_dir / "individual"
        indiv_out_dir.mkdir(exist_ok=True)
    else:
        indiv_out_dir = None

    num_prototypes = ppnet.num_prototypes
    prototype_class_identities = ppnet.prototype_class_identity.argmax(dim=1).numpy()
    num_prototypes_per_class = ppnet.prototype_class_identity.sum(dim=0).numpy().astype(int)
    prototype_info = np.load(str(img_dir / "bb195.npy"))
    prototype_classes = prototype_info[:, 5]

    all_train_imgs = []
    all_test_imgs = []
    all_test_ids = []
    prototypes = []
    # num_prototypes_per_class = 10
    for prototype_id in range(num_prototypes):
        # prototype_class = prototype_id // num_prototypes_per_class
        prototype_dir = train_dir / str(prototype_id)
        prototype = {
            "original": load_img(img_dir / f"prototype-img-original{prototype_id}.png"),
            "heatmap": load_img(img_dir / f"prototype-img-original_with_self_act{prototype_id}.png"),
            "patch": load_img(img_dir / f"prototype-img{prototype_id}.png"),
            "patch_in_orig": load_img(prototype_dir / f"prototype_in_original_pimg.png"),
            "class": prototype_classes[prototype_id],
            "class_name": class_names[prototype_classes[prototype_id]] if class_names is not None else prototype_id,
            "img_idx": prototype_info[prototype_id, 0]
        }
        prototypes.append(prototype)
        imgs, class_ids = load_most_similar_images(
            prototype_dir,
            n, class_names, prototype, prototype_id, indiv_out_dir,
            "train"
        )

        prototype_test_dir = test_dir / str(prototype_id)
        test_imgs, test_class_ids = load_most_similar_images(
            prototype_test_dir,
            n, class_names, prototype, prototype_id, indiv_out_dir,
            "test"
        )

        plt.close("all")

        all_train_imgs.append(imgs)
        all_test_imgs.append(test_imgs)
        all_test_ids.append(test_class_ids)

    all_train_class_ids = np.load(str(train_dir / "full_class_id.npy"))
    all_test_class_ids = np.stack(all_test_ids)
    num_classes = ppnet.num_classes

    df = get_weight_df(ppnet, class_names)
    plot_num_used_prototypes(df, out_dir)
    if args.non_zero:
        plot_weight_distributions(df.loc[df.weight != 0], out_dir)
    else:
        plot_weight_distributions(df, out_dir)
    for i in range(num_classes):
        if args.non_zero:
            df_subset = df.loc[(df["class"] == i) & (df["weight"] != 0)]
        else:
            df_subset = df.loc[(df["class"] == i) & df["class_prototype"]]
        prototypes_to_plot = df_subset.prototype.values
        weights = df_subset.weight.values

        plot_global_explanation_for_class(
            prototypes_to_plot,
            all_train_imgs,
            all_train_class_ids,
            prototypes,
            weights,
            n=n,
            class_names=class_names,
            savefig=out_dir / f"class-{i}_train_global_explanation.png"
        )
        plot_global_explanation_for_class(
            prototypes_to_plot,
            all_test_imgs,
            all_test_class_ids,
            prototypes,
            weights,
            n=n,
            class_names=class_names,
            savefig=out_dir / f"class-{i}_test_global_explanation.png"
        )
        plt.close("all")

    print("Done!")


def load_most_similar_images(prototype_dir, n, class_names, prototype, prototype_id, indiv_out_dir=None, data_type="train"):
    class_ids = np.load(str(prototype_dir / "class_id.npy"))
    imgs = load_imgs(prototype_dir, n)
    if indiv_out_dir is not None:
        if class_names is None:
            labels = class_ids
        else:
            labels = [class_names[i] for i in class_ids]
        plot_most_similar_images(imgs, prototype, n, labels,
                                 indiv_out_dir / f"prototype-{prototype_id}_top-{n}_{data_type}_images.png")

    return imgs, class_ids


def plot_num_used_prototypes(df: pd.DataFrame, save_dir: Path):
    num_used = pd.DataFrame(df.groupby("class").weight.count())
    num_used["positive"] = df.loc[df.weight > 0].groupby("class").weight.count()
    num_used["negative"] = df.loc[df.weight < 0].groupby("class").weight.count()

    if "class_name" in df.columns:
        class_names = df.groupby(["class"]).class_name.first()
        num_used["label"] = class_names
    else:
        num_used["label"] = num_used.index.astype(str)

    fig, ax = plt.subplots()
    ax.bar(num_used["label"], num_used["positive"], color="C0", label="Positive")
    ax.bar(num_used["label"], num_used["negative"], bottom=num_used["positive"], color="C1", label="Negative")
    plt.legend(frameon=False)
    plt.xlabel("Class")
    plt.ylabel("No. Prototypes")
    if "class_name" in df.columns:
        plt.xticks(rotation=45, horizontalalignment='right', rotation_mode="anchor")
    plt.tight_layout()
    plt.savefig(save_dir / "number_of_prototypes_used_per_class.png")


def plot_weight_distributions(df: pd.DataFrame, save_dir: Path):
    fig = sns.displot(df, x="weight", hue="class", kind="kde")
    plt.savefig(save_dir / "weight_distribution_by_class.png")

    if "class_name" in df.columns:
        fig = sns.displot(df, x="weight", hue="class_name", kind="kde")
        plt.savefig(save_dir / "weight_distribution_by_class_name.png")

    sns.displot(df, x="weight", hue="class_prototype", kind="kde")
    plt.xlim(fig.ax.get_xlim())
    plt.savefig(save_dir / "weight_distribution_by_class_prototype.png")

    sns.displot(df, x="weight", hue="class_prototype", kind="hist")
    plt.savefig(save_dir / "weight_distribution_by_class_prototype_hist.png")

    sns.displot(df, x="weight", hue="class_prototype", kind="hist", row="class")
    plt.xlim(fig.ax.get_xlim())
    plt.savefig(save_dir / "weight_distribution_by_class_prototype_per_class.png")


def get_weight_df(ppnet, class_names: list = None):
    prototype_class_identities = ppnet.prototype_class_identity.argmax(dim=1).numpy().astype(int)
    num_prototypes_per_class = ppnet.prototype_class_identity.sum(dim=0).numpy().astype(int)
    num_prototypes = ppnet.num_prototypes
    num_classes = ppnet.last_layer.weight.shape[0]
    same_class_weights = np.zeros(ppnet.last_layer.weight.shape)
    l_idx = 0
    for i in range(num_classes):
        u_idx = l_idx + num_prototypes_per_class[i]
        same_class_weights[i, l_idx:u_idx] = 1
        l_idx = u_idx
    data = [ppnet.last_layer.weight.detach().cpu().ravel().numpy(), np.arange(num_classes, dtype=int).repeat(num_prototypes), np.tile(np.arange(num_prototypes), num_classes), np.tile(prototype_class_identities, num_classes), same_class_weights.ravel()]
    data = np.array(data).T
    df = pd.DataFrame(data, columns=["weight", "class", "prototype", "prototype_class", "class_prototype"])
    int_cols = ["class", "prototype", "prototype_class", "class_prototype"]
    for col in int_cols:
        df[col] = df[col].astype(int)
    if class_names is not None:
        df["class_name"] = df["class"].map(lambda x: class_names[x])
    return df


def plot_global_explanation_for_class(prototype_ids, imgs, ids, prototypes, weights, n=10, class_names=None, savefig=None):
    imgs = [imgs[i] for i in prototype_ids]
    ids = ids[prototype_ids]
    prototypes = [prototypes[i] for i in prototype_ids]
    num_prototypes = ids.shape[0]
    img_type_dict = {
        0: "patch_in_orig",
        1: "heatmap"
    }
    n_rows = len(img_type_dict) * num_prototypes
    n_cols = n + 2
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2, n_rows*2))
    for row_num, row in enumerate(axes):
        img_type = row_num % len(img_type_dict)
        class_id = row_num // len(img_type_dict)
        for i, ax in enumerate(row):
            if i == 0:
                if img_type == 0:
                    add_text(ax, f"W: {weights[class_id]:.3f}\n"
                                 f"P: {prototype_ids[class_id]}")
            elif i == 1:
                ax.imshow(prototypes[class_id][img_type_dict[img_type]])
                if img_type == 0:
                    ax.set_title(prototypes[class_id]["class_name"])
            else:
                img_idx = i - 2
                ax.imshow(imgs[class_id][img_type_dict[img_type]][img_idx])
                if img_type == 0:
                    img_class_id = ids[class_id, img_idx]
                    if class_names is None:
                        title = img_class_id
                    else:
                        title = class_names[img_class_id]
                    ax.set_title(title)

    for ax in axes.flatten():
        remove_tick_labels(ax)
    plt.tight_layout()
    if savefig is not None:
        plt.savefig(savefig)
    else:
        plt.show()


def plot_most_similar_images(imgs, prototype, n=10, labels=None, savefig=None):
    n_axes = n+1
    fig, axes = plt.subplots(2, n_axes, figsize=(n_axes*2, 4))
    img_type_dict = {
        0: "patch_in_orig",
        1: "heatmap"
    }
    for row_num, img_type in img_type_dict.items():
        axes[row_num][0].imshow(prototype[img_type])
        if row_num == 0:
            axes[row_num][0].set_title(prototype["class_name"])
        for i in range(n):
            ax = axes[row_num][i+1]
            ax.imshow(imgs[img_type][i])
            if row_num == 0 and labels is not None:
                ax.set_title(labels[i])

    for ax in axes.flatten():
        remove_tick_labels(ax)
    plt.tight_layout()
    if savefig is not None:
        plt.savefig(savefig)
    else:
        plt.show()


def remove_tick_labels(ax):
    ax.xaxis.set_ticklabels([])
    ax.set_xticks([])
    ax.yaxis.set_ticklabels([])
    ax.set_yticks([])
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)


def load_img_type(name, ks):
    return [load_img(name.format(k)) for k in ks]


def load_imgs(img_dir, n=10):
    ks = list(range(1, n+1))
    img_type_dict = {
        "original": "nearest-{0}_original.png",
        "heatmap": "nearest-{0}_original_with_heatmap.png",
        "patch": "nearest-{0}_high_act_patch.png",
        "patch_in_orig": "nearest-{0}_high_act_patch_in_original_img.png"
    }
    imgs = {k: load_img_type(str(img_dir / v), ks) for k, v in img_type_dict.items()}
    return imgs


def load_img(path):
    return np.array(Image.open(path))


def add_text(ax, text, x=0.5, y=0.5, fontsize=20):
    kw = dict(ha="center", va="center", fontsize=fontsize)
    ax.text(x, y, text, transform=ax.transAxes, **kw)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model", help="Path to model .pth file. "
                                        "It is assumed the global_analysis.py output in in the same parent directory.")
    parser.add_argument("--n", type=int, default=10, help="Number of imgs to load per explanation.")
    parser.add_argument("--indiv", action="store_true", help="Plot individual figures for each prototype.")
    parser.add_argument("--non-zero", action="store_true",
                        help="Show all non-zero prototypes in explanation. Otherwise, only show class prototypes.")
    parser.add_argument("--class-names", default=None, help="Path to .txt containing \\n separated class names")
    args = parser.parse_args()
    main(args)
