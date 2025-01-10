
from pathlib import Path
from argparse import ArgumentParser
import re

import matplotlib.pyplot as plt
from matplotlib.transforms import Bbox
import numpy as np
from PIL import Image


def main(args):
    img_dir = Path(args.dir)
    orig_img = load_img(img_dir / "original_img.png")
    n = args.n

    with open(img_dir / "local_analysis.log", "r") as fp:
        log = fp.read()

    topk_meta = get_top_k_meta(log)
    most_activated_meta = get_most_activated_meta(log)

    most_activated_dir = img_dir / "most_activated_prototypes"
    load_and_plot_explanation(most_activated_dir, "most_activated", n, most_activated_meta, show_sum=False)

    proto_folders = [d for d in img_dir.glob("*") if d.is_dir()]
    topk_folders = [d.name for d in proto_folders if "top" in d.name]

    if len(topk_folders) != 0:
        topk = max([int(re.search("[0-9]+", d).group(0)) for d in topk_folders])

        for k in range(1, topk+1):
            top_dir = img_dir / f"top-{k}_class_prototypes"
            load_and_plot_explanation(top_dir, f"top-{k}", n, topk_meta[k-1]["meta"])
    else:
        meta = topk_meta[0]["meta"]
        all_dir = img_dir / "prototype_activations"
        load_and_plot_explanation(all_dir, "all_non_zero", len(meta), meta, show_sum=False)
    print("Done!")


def get_meta(log, most_activated=False):
    prototype_data = log.split("--------------------------------------------------------------")
    prototype_data = [v.strip() for v in prototype_data][:-1]
    prototype_data[0] = "\n".join(prototype_data[0].split("\n")[1:])
    if most_activated:
        prototype_data = [v.split("\n")[1:-3] for v in prototype_data]
    else:
        prototype_data = [v.split("\n")[:-3] for v in prototype_data]
    prototype_data = [[re.search(r"[-+]?(?:\d*\.\d+|\d+)", in_v).group(0) for in_v in v] for v in prototype_data]
    for i in range(len(prototype_data)):
        # prototype, class, (connection), similarity, fc weight
        if len(prototype_data[i]) == 4:
            prototype_data[i][0] = int(prototype_data[i][0])
            prototype_data[i][1] = int(prototype_data[i][1])
            prototype_data[i][2] = float(prototype_data[i][2])
            prototype_data[i][3] = float(prototype_data[i][3])
        elif len(prototype_data[i]) == 5:
            prototype_data[i][0] = int(prototype_data[i][0])
            prototype_data[i][1] = int(prototype_data[i][1])
            prototype_data[i][2] = float(prototype_data[i][3])
            prototype_data[i][3] = float(prototype_data[i][4])
        else:
            raise ValueError("Length of prototype data not recognised")
    return prototype_data


def get_top_k_meta(log):
    log = log.split("Prototypes from top-")[1]
    topk_logs = log.split("predicted class:")[1:]
    meta = []
    for topk_log in topk_logs:
        class_idx = int(topk_log.split("\n")[0].strip())
        logit = float(topk_log.split("\n")[1].split("logit of the class: ")[-1].strip())
        topk_log = "\n".join(topk_log.split("\n")[1:])
        meta.append({"class": class_idx, "logit": logit, "meta": get_meta(topk_log, most_activated=False)})
    return meta


def get_most_activated_meta(log):
    log = log.split("Most activated")[1]
    log = log.split("Prototypes from top-")[0]
    return get_meta(log, most_activated=True)


def load_and_plot_explanation(img_dir, name, n, meta, show_sum=True, dropzero=True):
    if dropzero:
        weights = [meta[i][3] for i in range(len(meta))]
        non_zero_idx = [i for i in range(len(weights)) if weights[i] != 0]
        meta2 = [meta[i] for i in non_zero_idx]
        meta = meta2
        n = min([n, len(meta)])
        indices = [non_zero_idx[i] for i in range(n)]
    else:
        indices = list(range(n))
    imgs = load_imgs(img_dir, indices=indices)
    prototypes = load_prototypes(img_dir, indices=indices)
    plot_explanation(imgs, prototypes, n, img_dir.parent / f"{name}_explanation.png", meta, show_sum=show_sum)


def add_text(ax, text, x=0.5, y=0.5, fontsize=20):
    kw = dict(ha="center", va="center", fontsize=fontsize)
    ax.text(x, y, text, transform=ax.transAxes, **kw)


def plot_explanation(imgs, prototypes, n=10, savefig=None, meta=None, show_sum=True):
    width = 14
    height = 1 + 1.5*n
    width_ratios = [1 for i in range(3)] + [2 for i in range(6)]
    class_names = ["13-16", "16-18", "18-20", "20-22", "22-24"]
    classes = [class_names[meta[i][1]] for i in range(n)]
    fig, axes = plt.subplots(n, 9, figsize=(width, height), gridspec_kw={'width_ratios': width_ratios})
    titles = ["Similarity", "Weight", "Contribution", "Prototype", "Proto patch", "Proto heatmap", "Img crop", "Img patch", "Img heatmap"]
    for i, row in enumerate(axes):
        #row[0].set_ylabel(f"{meta[i][1]}", rotation='horizontal', ha='right', fontsize=20, labelpad=20)
        add_text(row[0], f"{meta[i][2]:.3f}")
        add_text(row[1], f"{meta[i][3]:.3f}")
        add_text(row[2], f"{meta[i][2] * meta[i][3]:.3f}")
        row[3].imshow(prototypes["patch"][i])
        #row[3].set_xlabel(f"{meta[i][0]}", fontsize=15)
        row[4].imshow(prototypes["patch_in_orig"][i])
        row[5].imshow(prototypes["heatmap"][i])
        row[6].imshow(imgs["patch"][i])
        row[7].imshow(imgs["patch_in_orig"][i])
        row[8].imshow(imgs["heatmap"][i])

        if i == 0:
            for j, title in enumerate(titles):
                if title == "Prototype":
                    row[j].set_title(title, y=1.0, pad=11)
                else:
                    row[j].set_title(title)
    if show_sum:
        add_text(axes[-1][-1], f"{sum([meta[i][2] * meta[i][3] for i in range(len(axes))]):.3f}", y=0)
    for ax in axes.flatten():
        #ax.axis("off")
        ax.xaxis.set_ticklabels([])
        ax.set_xticks([])
        ax.yaxis.set_ticklabels([])
        ax.set_yticks([])
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
    #plt.tight_layout()
    if savefig is not None:
        # box = fig.bbox_inches
        #box = Bbox.from_bounds(1.0, 1.0, width - 2, height - 2)
        plt.tight_layout()
        plt.savefig(savefig, dpi=200)#, bbox_inches=box)
    else:
        plt.show()

"""
def plot_explanation(imgs, prototypes, n=10, savefig=None, meta=None, show_sum=True):
    width = 14
    height = 1 + 1.5*n
    fig, axes = plt.subplots(n, 9, figsize=(width, height))
    titles = ["Proto heatmap", "Proto patch", "Prototype","Img heatmap", "Img patch", "Img crop", "Similarity", "Weight", "Contribution"]
    for i, row in enumerate(axes):
        row[0].set_ylabel(f"{meta[i][1]}", rotation='horizontal', ha='right', fontsize=20, labelpad=20)
        row[0].imshow(prototypes["heatmap"][i])
        row[1].imshow(prototypes["patch_in_orig"][i])
        row[2].imshow(prototypes["patch"][i])
        row[3].imshow(imgs["heatmap"][i])
        row[4].imshow(imgs["patch_in_orig"][i])
        row[5].imshow(imgs["patch"][i])
        add_text(row[6], f"{meta[i][2]:.3f}")
        add_text(row[7], f"{meta[i][3]:.3f}")
        add_text(row[8], f"{meta[i][2] * meta[i][3]:.3f}")
        if i == 0:
            for j, title in enumerate(titles):
                row[j].set_title(title)
    if show_sum:
        add_text(axes[-1][-1], f"{sum([meta[i][2] * meta[i][3] for i in range(len(axes))]):.3f}", y=0)
    for ax in axes.flatten():
        #ax.axis("off")
        ax.xaxis.set_ticklabels([])
        ax.set_xticks([])
        ax.yaxis.set_ticklabels([])
        ax.set_yticks([])
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
    #plt.tight_layout()
    if savefig is not None:
        # box = fig.bbox_inches
        #box = Bbox.from_bounds(1.0, 1.0, width - 2, height - 2)
        plt.tight_layout()
        plt.savefig(savefig)#, bbox_inches=box)
    else:
        plt.show()
"""


def load_prototypes(img_dir, n=10, indices=None):
    if indices is None:
        indices = list(range(n))
    imgs = {}
    imgs["heatmap"] = [load_img(img_dir / f"top-{i+1}_activated_prototype_self_act.png")
                       for i in indices]
    imgs["patch"] = [load_img(img_dir / f"top-{i+1}_activated_prototype.png")
                     for i in indices]
    imgs["patch_in_orig"] = [load_img(img_dir / f"top-{i+1}_activated_prototype_in_original_pimg.png")
                             for i in indices]
    return imgs


def load_imgs(img_dir, n=10, indices=None):
    if indices is None:
        indices = list(range(n))
    imgs = {}
    imgs["heatmap"] = [load_img(img_dir / f"prototype_activation_map_by_top-{i+1}_prototype.png")
                       for i in indices]
    imgs["patch"] = [load_img(img_dir / f"most_highly_activated_patch_by_top-{i+1}_prototype.png")
                     for i in indices]
    imgs["patch_in_orig"] = [load_img(img_dir / f"most_highly_activated_patch_in_original_img_by_top-{i+1}_prototype.png")
                             for i in indices]
    return imgs


def load_img(path):
    return np.array(Image.open(path))


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--dir", help="Directory containing local_analysis.py output")
    parser.add_argument("--n", type=int, default=10, help="Number of imgs to load per explanation")
    args = parser.parse_args()
    main(args)
