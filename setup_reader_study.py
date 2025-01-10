import shutil
from argparse import ArgumentParser
from pathlib import Path
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import cv2

from helpers import find_high_activation_crop
from get_dataset_results import create_dataloader


def main(args):
    plt.rcParams.update({'font.size': 14})
    n = args.n
    results_dir = Path(args.results)
    img_dir = Path(args.img_dir)
    assert img_dir.exists()

    reader_study_dir = results_dir / "reader_study"
    intruder_dir = reader_study_dir / "intruder"
    intruder_dir.mkdir(exist_ok=True, parents=True)
    img_types_to_plot = ["bbox"] #, "overlay", "patch"]
    for img_type in img_types_to_plot:
        out_dir = intruder_dir / img_type
        out_dir.mkdir(exist_ok=True)
    concept_discovery_dir = reader_study_dir / "concept_discovery"
    binary_choice_dir = reader_study_dir / "binary_choice"
    utility_dir = reader_study_dir / "utility"
    utility_pred_dir = utility_dir / "pred"
    utility_explain_dir = utility_dir / "explain"
    utility_plain_dir = utility_dir / "plain"
    for d in [concept_discovery_dir, binary_choice_dir, utility_dir, utility_pred_dir, utility_plain_dir, utility_explain_dir]:
        d.mkdir(exist_ok=True)

    via_dir = Path(args.via_dir)
    via_json_names = [
        "intruder",
        "concept_discovery",
        "binary_choice",
        "utility_plain",
        "utility_pred",
        "utility_explanation",
    ]

    jsons = {}
    for json_name in via_json_names:
        with open(via_dir / f"{json_name}.json", "r") as fp:
            jsons[json_name] = json.load(fp)

    model_path = Path(args.model)
    model = torch.load(model_path).cpu()
    model_dir = model_path.parent
    model_epoch = int(model_path.name.split("_")[0])
    model_proto_dir = model_dir / "img" / f"epoch-{model_epoch}"
    train_dataloader = create_dataloader(args.train_dir, shuffle=False, normalise=False)
    with open("/home/lina3782/labs/explain/intergrowth/processed/metadata.json", "r") as fp:
        intergrowth_meta = json.load(fp)
    intergrowth_meta = intergrowth_meta["test"]

    prototype_info = np.load(str(model_proto_dir / f"bb{model_epoch}.npy"))  # P, 6
    similaritites = np.load(str(results_dir / "similarities.npy"))  # N, P
    similarity_maps = np.load(str(results_dir / "similarity_maps.npy"))  # N, P, H, W (H=W=7)
    contributions = np.load(str(results_dir / "contributions.npy"))  # N, P, C

    with open(results_dir / "results.json", "r") as fp:
        results = json.load(fp)

    with open(args.class_names, "r") as fp:
        class_names = fp.read().split("\n")

    for i in range(len(results["img_results"])):
        metadata = intergrowth_meta[results["img_results"][i]["path"][:-4]]
        results["img_results"][i]["age"] = metadata["age"]
        results["img_results"][i]["patient"] = metadata["patient"]

    prototype_train_paths = [Path(train_dataloader.dataset.imgs[i][0]) for i in prototype_info[:, 0]]
    prototype_names = [v.name for v in prototype_train_paths]
    prototype_paths = [img_dir / name for name in prototype_names]

    # TODO: Add indices to exclude
    n_images_per_class_utility = 5
    utility_results = []
    utility_results_indices = []
    result_class_indices = [[i for i in range(len(results["img_results"])) if results["img_results"][i]["target"] == k] for k in range(model.num_classes)]
    rng = np.random.default_rng(66642)
    for k in range(model.num_classes):
        sample_indices = rng.choice(result_class_indices[k], n_images_per_class_utility, replace=False)
        sample = [results["img_results"][i] for i in sample_indices]
        utility_results = sample + utility_results
        utility_results_indices = sample_indices.tolist() + utility_results_indices

    sns.displot([v["age"] for v in utility_results], bins=np.arange(11, 43) + 0.5, aspect=1.4)
    plt.xlabel("Gestational Age / Weeks")
    plt.tight_layout()
    plt.show()

    # Create utility VIA files
    utility_results_meta_keys = ["path", "patient", "age", "target", "prediction", "correct"]
    utility_results_meta = [{k: v for k, v in result.items() if k in utility_results_meta_keys} for result in utility_results]
    utility_files = [img_dir / result["path"] for result in utility_results]

    for i, result in enumerate(utility_results):
        result["idx"] = utility_results_indices[i]
        result["similarity_map"] = similarity_maps[result["idx"]]
        img = cv2.imread(str(img_dir / result["path"]))
        plot_utility_plain(img, utility_plain_dir / result["path"])
        plot_utility_pred(img, class_names[result["prediction"]], utility_pred_dir / result["path"])
        result_contributions = contributions[result["idx"]][:, result["prediction"]]
        top_p = result_contributions.argsort()[::-1][:4]
        utility_results_meta[i]["top_prototypes"] = top_p.tolist()
        proto_images = [get_prototype_imgs(prototype_paths, model_proto_dir, p, prototype_info[p, 1:5]) for p in top_p]
        result_images = [load_and_process_img_from_result(result, p, img_dir) for p in top_p]
        plot_explanation_with_orig(result_images, proto_images, img, class_names[result["prediction"]], savefig=utility_explain_dir / result["path"])

    utility_df = pd.DataFrame(utility_results_meta)
    utility_plain_files = [utility_plain_dir / result["path"] for result in utility_results]
    create_exp_outputs(jsons["utility_plain"], utility_plain_files, rng, utility_df, via_dir, utility_plain_dir, "utility_plain")

    utility_df = pd.DataFrame(utility_results_meta)
    utility_pred_files = [utility_pred_dir / result["path"] for result in utility_results]
    create_exp_outputs(jsons["utility_pred"], utility_pred_files, rng, utility_df, via_dir, utility_pred_dir, "utility_pred")

    utility_df = pd.DataFrame(utility_results_meta)
    utility_explain_files = [utility_explain_dir / result["path"] for result in utility_results]
    create_exp_outputs(jsons["utility_explanation"], utility_explain_files, rng, utility_df, via_dir, utility_explain_dir, "utility_explanation")

    # Now I need to work out how to get the prompts out of the way
    # Work out what subset of the birds I'm going to get done for lab mates

    # Write instructions to send with the study
    # Make sure the software saves the timings
    # Send an example study to Liz, with ~ all 3 steps, so we can time how long it takes

    print(f"N, P: {similaritites.shape}")
    n_prototypes = similaritites.shape[1]
    n_for_array = 2*n
    top_n_inds = np.zeros((n_prototypes, n_for_array), dtype=int)
    top_n_results = []
    for p in range(n_prototypes):
        top_n = np.argsort(similaritites[:, p])[-n_for_array:][::-1]
        top_n_inds[p] = top_n
        sub_results = [results["img_results"][i] for i in top_n]
        for i, idx in enumerate(top_n):
            sub_results[i]["idx"] = idx
            sub_results[i]["similarity_map"] = similarity_maps[idx]
        top_n_results.append(sub_results)

    top_n_results_by_class_ind = [[] for i in range(len(class_names))]
    for p in range(n_prototypes):
        for i in range(len(top_n_results[p])):
            top_n_results_by_class_ind[top_n_results[p][i]["target"]].append((p, i))

    rng = np.random.default_rng(42)
    intruder_exp_meta = {"indices": [], "random": []}
    binary_exp_meta = {"indices": [], "random_indices": [], "correct_left": []}
    concept_exp_meta = {k: {"indices": []} for k in ["similar", "random", "class", "class_range"]}
    concept_exp_meta["class"]["class"] = []
    concept_exp_meta["class_range"]["range"] = []
    for p in range(n_prototypes):
        allowed_random_p = list(range(n_prototypes))
        allowed_random_p.pop(p)

        # Load images
        processed_imgs = load_n_images(top_n_results, n, p, img_dir)

        # Load prototype
        proto_result = get_prototype_imgs(prototype_paths, model_proto_dir, p, prototype_info[p, 1:5])
        proto_class = prototype_info[p, -1]
        proto_class_name = class_names[proto_class]

        # Plot concept discovery experiment
        img_order = list(range(len(processed_imgs) + 1))[1:]
        rng.shuffle(img_order)
        plot_concept_discovery_exp([proto_result] + processed_imgs, [0] + img_order, concept_discovery_dir / f"prototype_{p:04d}.png")
        concept_exp_meta["similar"]["indices"].append(get_img_idx(processed_imgs, img_order[:7], minus=1))

        # Plot random concept_discovery_experiment
        random_images = load_n_random_images(top_n_results, allowed_random_p, 7, rng, img_dir)
        img_order = list(range(len(random_images) + 1))[1:]
        rng.shuffle(img_order)
        plot_concept_discovery_exp([proto_result] + random_images, [0] + img_order, concept_discovery_dir / f"prototype_{p:04d}_random.png")
        concept_exp_meta["random"]["indices"].append(get_img_idx(random_images, img_order, minus=1))

        # Plot random from same class concept discovery experiment
        selected_images = load_n_images_from_class(top_n_results, top_n_results_by_class_ind, p, [proto_class], 7, rng, img_dir)
        img_order = list(range(len(selected_images) + 1))[1:]
        rng.shuffle(img_order)
        plot_concept_discovery_exp([proto_result] + selected_images, [0] + img_order, concept_discovery_dir / f"prototype_{p:04d}_same_class.png")
        concept_exp_meta["class"]["indices"].append(get_img_idx(selected_images, img_order, minus=1))
        concept_exp_meta["class"]["class"].append(proto_class)

        # Plot random from nearby classes concept discovery experiment
        allowed_classes = get_class_range(rng, proto_class, class_names)
        selected_images = load_n_images_from_class(top_n_results, top_n_results_by_class_ind, p, allowed_classes, 7, rng, img_dir)
        img_order = list(range(len(selected_images) + 1))[1:]
        rng.shuffle(img_order)
        plot_concept_discovery_exp([proto_result] + selected_images, [0] + img_order, concept_discovery_dir / f"prototype_{p:04d}_nearby_class.png")
        concept_exp_meta["class_range"]["indices"].append(get_img_idx(selected_images, img_order, minus=1))
        concept_exp_meta["class_range"]["range"].append(allowed_classes)

        """
        random_p = rng.choice(allowed_random_p)
        random_proto_images = load_n_images(top_n_results, n, random_p, img_dir)
        rand_img_order = list(range(len(random_proto_images)))
        """

        # Plot binary choice
        random_images = load_n_random_images(top_n_results, allowed_random_p, n, rng, img_dir)
        img_order = list(range(len(processed_imgs)))
        left = rng.integers(2)
        if left:
            plot_binary_choice(processed_imgs, random_images, left_indices=img_order, savefig=binary_choice_dir / f"prototype_{p:04d}.png")
        else:
            plot_binary_choice(random_images, processed_imgs, right_indices=img_order, savefig=binary_choice_dir / f"prototype_{p:04d}.png")
        binary_exp_meta["correct_left"].append(bool(left))
        binary_exp_meta["random_indices"].append([v["idx"] for v in random_images])
        binary_exp_meta["indices"].append(get_img_idx(processed_imgs, img_order))

        # Only keep the closest 5 images
        processed_imgs = processed_imgs[:5]
        for output in processed_imgs:
            output["random"] = False

        # Choose a random image to be the odd one out
        random_p = rng.choice(allowed_random_p)
        random_idx = rng.choice(n)
        processed_imgs.append(
            load_and_process_img_from_result(top_n_results[random_p][random_idx], random_p, img_dir)
        )
        processed_imgs[-1]["random"] = True

        # Plot the image intruder experiment
        img_order = list(range(6))
        rng.shuffle(img_order)
        for img_type in img_types_to_plot:
            out_path = intruder_dir / img_type / f"prototype_{p}.png"
            plot_intruder_exp(processed_imgs, img_order, img_type=img_type, savefig=out_path)
        idx = [v["idx"] for v in processed_imgs]
        intruder_exp_meta["indices"].append([idx[i] for i in img_order])
        random_loc = np.where(np.array([v["random"] for v in processed_imgs])[img_order])[0][0]
        intruder_exp_meta["random"].append(random_loc)

        plt.close("all")

    # Create intruder VIA files
    intruder_files = [reader_study_dir / f"intruder/bbox/prototype_{p}.png" for p in range(n_prototypes)]
    intruder_df = pd.DataFrame(intruder_exp_meta["indices"])
    intruder_df["random"] = intruder_exp_meta["random"]
    create_exp_outputs(jsons["intruder"], intruder_files, rng, intruder_df, via_dir, intruder_dir, "intruder")

    # Create concept VIA files
    concept_endings = ["", "_nearby_class", "_random", "_same_class"]
    concept_types = ["similar", "class_range", "random", "class"]
    concept_files = [concept_discovery_dir / f"prototype_{p:04d}{end}.png" for end in concept_endings for p in range(n_prototypes)]
    concept_dfs = []
    for v in concept_types:
        df = pd.DataFrame(concept_exp_meta[v]["indices"])
        df["img_type"] = v
        df["prototype"] = range(n_prototypes)
        if v == "class":
            df["class"] = concept_exp_meta["class"]["class"]
        elif v == "class_range":
            df["class_range"] = concept_exp_meta["class_range"]["range"]
        concept_dfs.append(df)
    concept_df = pd.concat(concept_dfs, axis=0)
    concept_df = concept_df.reset_index(drop=True)
    create_exp_outputs(jsons["concept_discovery"], concept_files, rng, concept_df, via_dir, concept_discovery_dir, "concept")

    # Create binary VIA files
    binary_files = [binary_choice_dir / f"prototype_{p:04d}.png" for p in range(n_prototypes)]
    binary_df = pd.DataFrame(binary_exp_meta["indices"], columns=[f"concept_{i}" for i in range(10)])
    binary_df2 = pd.DataFrame(binary_exp_meta["random_indices"], columns=[f"random_{i}" for i in range(10)])
    binary_df = pd.concat([binary_df, binary_df2], axis=1)
    binary_df["correct_left"] = binary_exp_meta["correct_left"]
    create_exp_outputs(jsons["binary_choice"], binary_files, rng, binary_df, via_dir, binary_choice_dir, "binary")


    print("Done!")


def plot_utility_plain_old(img, savefig):
    fig, ax = plt.subplots()
    ax.imshow(img)
    ax.axis("off")
    plt.tight_layout()
    plt.subplots_adjust(top=0.80, bottom=0.05)
    plt.savefig(savefig)
    plt.close(fig)

def plot_utility_plain(img, savefig):
    white_space = np.ones((150, 781, 3), dtype=np.uint8) * 255
    out_img = np.concatenate((white_space, img))
    cv2.imwrite(savefig, out_img)

def plot_utility_pred(img, label, savefig):
    fig, ax = plt.subplots()
    ax.imshow(img)
    ax.axis("off")
    ax.set_title(label)
    plt.tight_layout()
    plt.subplots_adjust(top=0.80, bottom=0.05)
    plt.savefig(savefig)
    plt.close(fig)


def plot_explanation(result_images, proto_images, savefig=None):
    images = [proto_images, result_images]
    n = len(result_images)
    width = 8
    height = 1 + 1.5*n
    fig = plt.figure(figsize=(width, height), constrained_layout=True)
    subfigs = fig.subfigures(1, 2)
    titles = ["Prototype", "Test Image"]
    plot_explanation_on_subfigs(subfigs, images, n, titles)
    plt.savefig(savefig, dpi=200)
    plt.close(fig)


def plot_explanation_on_subfigs(subfigs, images, n, titles):
    for i, subfig in enumerate(subfigs):
        axes = subfig.subplots(n, 2)
        for j, row in enumerate(axes):
            row[0].imshow(images[i][j]["bbox"])
            row[1].imshow(images[i][j]["overlay"])

        for ax in axes.flatten():
            remove_clutter(ax)

        subfig.suptitle(titles[i])


def plot_explanation_with_orig(result_images, proto_images, orig_img, label=None, savefig=None):
    images = [proto_images, result_images]
    n = len(result_images)
    width = 16
    height = 1 + 1.5 * n
    fig = plt.figure(figsize=(width, height), constrained_layout=True)
    subfigs = fig.subfigures(1, 2)
    explain_subfigs = subfigs[1].subfigures(1, 2)
    titles = ["Explanation", "Test Image"]
    plot_explanation_on_subfigs(explain_subfigs, images, n, titles)
    axes = subfigs[0].subplots(2, 1, gridspec_kw={'height_ratios': [1, 6]})
    ax = axes[1]
    ax.imshow(orig_img)
    ax.set_title(label)
    for ax in axes:
        remove_clutter(ax)
    plt.savefig(savefig, dpi=200)
    plt.close(fig)


def remove_clutter(ax):
    ax.xaxis.set_ticklabels([])
    ax.set_xticks([])
    ax.yaxis.set_ticklabels([])
    ax.set_yticks([])
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)


def create_exp_outputs(template, files, rng, df, via_dir, out_dir, exp_name):
    out_order, outfiles = randomise_output_order(len(files), rng, exp_name)
    exp_json = create_exp_json(template, outfiles)

    outfiles_ordered = [outfiles[i] for i in out_order]
    df["file"] = files
    df["outfile"] = outfiles_ordered
    df["out_idx"] = out_order
    df.to_csv(out_dir / "img_indices.csv", index=False)
    # out_order_inverse = [out_order.index(i) for i in range(len(files))]
    # df["outfile_idx"] = out_order_inverse

    (via_dir / exp_name).mkdir(exist_ok=True, parents=True)
    for i in range(len(files)):
        shutil.copy(files[i], via_dir / outfiles_ordered[i])

    with open(via_dir / f"{exp_name}.json", "w") as fp:
        json.dump(exp_json, fp, indent=2)


def randomise_output_order(n_files, rng, exp_name):
    output_order = list(range(n_files))
    rng.shuffle(output_order)
    outfiles = [f"./{exp_name}/{i:04d}.png" for i in range(n_files)]
    return output_order, outfiles


def create_exp_json(template, outfiles):
    n_files = len(outfiles)
    file_dict = {str(i): {"fid": str(i), "fname": outfiles[i], "type": 2, "loc": 3, "src": outfiles[i]} for i in range(n_files)}
    view_dict = {str(i): {"fid_list": [str(i)]} for i in range(n_files)}
    template["file"] = file_dict
    template["view"] = view_dict
    template["project"]["vid_list"] = [str(i) for i in range(n_files)]
    return template


def get_img_idx(imgs, img_order, minus=0):
    img_idx = [v["idx"] for v in imgs]
    return [img_idx[i-minus] for i in img_order]


def get_class_range(rng, proto_class, class_names):
    size = rng.integers(2) + 2
    l = r = proto_class
    if size == 2:
        left = rng.choice(2)
        if left:
            l += -1
        else:
            r += 1
    else:
        l += -1
        r += 1
    shift = rng.integers(3) - 1
    l += shift
    r += shift

    if l < 0:
        add = -l
        r += add
        l += add
    if r >= len(class_names):
        add = -(r - len(class_names) + 1)
        r += add
        l += add

    allowed_classes = np.arange(l, r + 1)
    return allowed_classes.tolist()


def plot_binary_choice(left_imgs, right_imgs, left_indices=None, right_indices=None, savefig=None):
    if left_indices is None:
        left_indices = list(range(len(left_imgs)))
    if right_indices is None:
        right_indices = list(range(len(right_imgs)))

    left_idx = [0, 1, 2, 7, 8, 9]
    left_i = 0
    right_i = 0
    width_ratios = [2 for _ in range(3)] + [1] + [2 for _ in range(3)]
    fig, axes = plt.subplots(2, 7, figsize=(16, 4), gridspec_kw={'width_ratios': width_ratios})
    axes = axes.flatten()
    for i, ax in enumerate(axes):
        if i == 3 or i == 10:
            pass
        else:
            if i in left_idx:
                img = left_imgs[left_indices[left_i]]["bbox"]
                left_i += 1
            else:
                img = right_imgs[right_indices[right_i]]["bbox"]
                right_i += 1
            ax.imshow(img)
        ax.axis("off")
    plt.tight_layout()
    plt.subplots_adjust(top=0.75, bottom=0.05)

    if savefig is None:
        plt.show()
    else:
        plt.savefig(savefig)


def plot_concept_discovery_exp(processed_imgs, indices=None, savefig=None):
    if indices is None:
        indices = [0] + list(range(1, len(processed_imgs)))

    img_types = {0: "bbox", 1: "overlay"}
    fig, axes = plt.subplots(4, 4, figsize=(16, 12))
    for row_idx, row in enumerate(axes):
        for col_idx, ax in enumerate(row):
            idx = 4*(row_idx > 1) + col_idx
            img_type_idx = row_idx % 2
            img = processed_imgs[indices[idx]][img_types[img_type_idx]]
            ax.imshow(img)
            ax.axis("off")
            if row_idx == 0 and col_idx == 0:
                ax.set_title("Prototype")
    plt.tight_layout()
    plt.subplots_adjust(top=0.85, bottom=0.05)
    if savefig is None:
        plt.show()
    else:
        plt.savefig(savefig)


def plot_intruder_exp(processed_imgs, indices=None, img_type="bbox", savefig=None):
    if indices is None:
        indices = range(len(processed_imgs))
    fig, axes = plt.subplots(2, 3, figsize=(12, 7.5))
    axes = axes.flatten()
    for i, ax in enumerate(axes):
        img = processed_imgs[indices[i]][img_type]
        ax.imshow(img)
        ax.axis("off")
        ax.set_title(i+1)
    plt.subplots_adjust(top=0.75, left=0.02, right=0.98, bottom=0.05)
    if savefig is None:
        plt.show()
    else:
        plt.savefig(savefig)


def load_n_images(top_n_results, n, p, img_dir) -> list:
    # Check for near duplicates and do not show both
    processed_imgs = []
    for result in top_n_results[p]:
        if len(processed_imgs) == n:
            break
        imgs = load_and_process_img_from_result(result, p, img_dir)
        if check_if_duplicate(imgs, processed_imgs):
            # print(f"Duplicate for prototype {p}")
            continue
        else:
            processed_imgs.append(imgs)
    return processed_imgs


def load_n_images_from_class(top_n_results, top_n_results_by_class_ind, p, classes, n, rng, img_dir) -> list:
    allowed_inds = []
    for i in classes:
        allowed_inds += top_n_results_by_class_ind[i]
    allowed_results = [top_n_results[i][j] for i, j in allowed_inds if i != p]
    selected_results = list(rng.choice(allowed_results, n, replace=False))
    selected_images = [load_and_process_img_from_result(result, p, img_dir) for result in selected_results]
    return selected_images


def load_n_random_images(top_n_results, allowed_random_p, n, rng, img_dir) -> list:
    random_images = []
    for i in range(n):
        random_p = rng.choice(allowed_random_p)
        idx = rng.integers(len(top_n_results[random_p]))
        random_images.append(load_and_process_img_from_result(top_n_results[random_p][idx], random_p, img_dir))
    return random_images


def check_if_duplicate(img, img_list, thresh=0.9):
    indices_to_check = list(range(len(img_list)))
    same_shape = [img_list[i]["img"].shape == img["img"].shape for i in indices_to_check]
    if sum(same_shape) > 0:  # Potential duplicates
        # print(f"Same shape > 1, {idx} and {i}")
        for i in np.where(same_shape)[0]:
            same_pix = (img_list[i]["img"] == img["img"]).sum() / img["img"].size
            duplicate = same_pix > thresh
            if duplicate:
                return True
    return False


def remove_duplicates(img_list, thresh=0.90):
    idx_to_remove = []
    for idx in range(len(img_list)):
        indices_to_check = list(range(idx+1, len(img_list)))
        same_shape = [img_list[i]["img"].shape == img_list[idx]["img"].shape for i in indices_to_check]
        if sum(same_shape) > 0:  # Potential duplicates
            # print(f"Same shape > 1, {idx} and {i}")
            for i in np.where(same_shape)[0]:
                idx2 = indices_to_check[i]
                if idx2 != idx:
                    same_pix = (img_list[idx2]["img"] == img_list[idx]["img"]).sum() / img_list[idx]["img"].size
                    duplicate = same_pix > thresh
                    if duplicate:
                        idx_to_remove.append(idx2)
    for idx in idx_to_remove:
        del img_list[idx]
    return img_list


def get_prototype_imgs(prototype_paths, model_proto_dir, p, bbox_info):
    bbox_info = bbox_info.copy()
    prototype = cv2.imread(str(prototype_paths[p]))
    proto_heatmap_ = cv2.imread(str(model_proto_dir / f"prototype-img-original_with_self_act{p}.png"))
    proto_heatmap_ = cv2.cvtColor(np.uint8(proto_heatmap_), cv2.COLOR_RGB2BGR)
    proto_heatmap = cv2.resize(proto_heatmap_, (prototype.shape[1], prototype.shape[0]))
    ratios = [prototype.shape[i]/proto_heatmap_.shape[i] for i in range(2)]
    # h0, h1, w0, w1
    bbox_info[:2] = bbox_info[:2] * ratios[0]
    bbox_info[2:] = bbox_info[2:] * ratios[1]
    bbox_overlay = add_bbox(prototype, bbox_info)
    h0, h1, w0, w1 = bbox_info
    patch = prototype[h0:h1, w0:w1]
    result = {
        "id": p,
        "path": str(prototype_paths[p]),
        "img": prototype,
        "bbox": bbox_overlay,
        "overlay": proto_heatmap,
        "patch": patch
    }
    return result


"""
.astype(float) / 255
        img = np.random.normal(0.02, 0.8,  img.shape) + img
        img[img > 1] = 1.0
        img[img < 0] = 0.0
        img = cv2.GaussianBlur(img, (11, 11), cv2.BORDER_DEFAULT)
        img[img > 1] = 1.0
"""


def load_and_process_img_from_result(result, p, img_dir):
    img = cv2.imread(str(img_dir / result["path"]))
    output = process_img(img, result["similarity_map"][p])
    output["idx"] = result["idx"]
    return output


def process_img(img, similarity_map):
    """Generate all versions of the explanation"""
    heatmap, rescaled_similarity_map = get_heatmap(similarity_map, (img.shape[:2]))
    overlay = get_overlay(img, similarity_map)
    bbox_info = find_high_activation_crop(rescaled_similarity_map)
    bbox_overlay = add_bbox(img, bbox_info)
    h0, h1, w0, w1 = bbox_info
    patch = img[h0:h1, w0:w1]

    output = {
        "img": img,
        "patch": patch,
        "bbox": bbox_overlay,
        "overlay": overlay
    }
    return output


"""
def plot_all():
    fig, axes = plt.subplots(1, 6, figsize=(12, 3))
    axes[0].imshow(high_res_img)
    axes[1].imshow(overlay)
    axes[2].imshow(patch)
    axes[3].imshow(bbox_overlay)
    axes[4].imshow(heatmap)
    axes[5].imshow(result["similarity_map"][p])
    for ax in axes:
        ax.axis("off")
    plt.tight_layout()
    plt.show()

prototype = cv2.imread(str(prototype_paths[p]))
proto_heatmap_ = cv2.imread(str(model_proto_dir / f"prototype-img-original_with_self_act{p}.png"))
proto_heatmap_ = cv2.cvtColor(np.uint8(proto_heatmap_), cv2.COLOR_RGB2BGR)
proto_heatmap = cv2.resize(proto_heatmap_, (prototype.shape[1], prototype.shape[0]))

fig, axes = plt.subplots(2, 2)
axes = axes.flatten()
axes[0].imshow(train_dataloader.dataset[prototype_info[p, 0]][0].permute(1, 2, 0))
axes[1].imshow(prototype)
axes[2].imshow(proto_heatmap_)
axes[3].imshow(proto_heatmap)
for ax in axes:
    ax.axis("off")
plt.tight_layout()
plt.show()
"""


def add_bbox(img, bbox_info, color=(0, 255, 255)):
    h0, h1, w0, w1 = bbox_info
    img_bgr_uint8 = cv2.cvtColor(np.uint8(img), cv2.COLOR_RGB2BGR)
    cv2.rectangle(img_bgr_uint8, (w0, h0), (w1-1, h1-1),
                  color, thickness=2)
    img_rgb_uint8 = img_bgr_uint8[...,::-1]
    img_rgb_float = np.float32(img_rgb_uint8) / 255
    return img_rgb_float


def get_heatmap(similarity_map, shape=None):
    shape = (shape[1], shape[0])
    upsampled_act_pattern = cv2.resize(similarity_map,
                                       dsize=shape,
                                       interpolation=cv2.INTER_CUBIC)
    rescaled_act_pattern = upsampled_act_pattern - np.amin(upsampled_act_pattern)
    rescaled_act_pattern = rescaled_act_pattern / np.amax(rescaled_act_pattern)
    heatmap = cv2.applyColorMap(np.uint8(255 * rescaled_act_pattern), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    heatmap = heatmap[..., ::-1]
    return heatmap, rescaled_act_pattern


def get_overlay(img, similarity_map):
    heatmap, rescaled_sim_map = get_heatmap(similarity_map, img.shape[:2])
    overlayed_img = 0.5 * img.astype(float) / 255 + 0.2 * heatmap
    return overlayed_img


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--results', help="Path to the output folder of get_dataset_results.py")
    parser.add_argument("--img-dir", help="Path to full resolution images")
    parser.add_argument("--train-dir", help="Path to train dataset used for this model")
    parser.add_argument("--model", help="Path to model .pth")
    parser.add_argument("--class-names", default=None, help="Path to .txt containing \\n separated class names")
    parser.add_argument("--via-dir", help="Path to directory containing via experiment .jsons")
    parser.add_argument("--n", default=10, type=int, help="No. most similar images for each prototype to generate")
    args = parser.parse_args()
    main(args)
