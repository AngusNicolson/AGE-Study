
import os
from argparse import ArgumentParser
from pathlib import Path

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from trainer import Pruner
from get_dataset_results import create_dataloader


def main(model_path, tau_range, tau_step=0.05, min_weights_range=(0, 1), dataset=None, workers=4):
    model_path = Path(model_path)

    ppnet = torch.load(model_path).cpu()

    taus = np.arange(*tau_range, tau_step)
    min_weights_values = np.arange(*min_weights_range).astype(int)

    combos = np.meshgrid(taus, min_weights_values)
    combos = np.array(combos).T.reshape(-1, 2)
    weights = ppnet.last_layer.weight.detach().numpy()
    num_weights = weights.size
    num_prototypes = weights.shape[1]

    if dataset is not None:
        dataloader = create_dataloader(dataset, ppnet.img_size, 128, False, workers)
        ppnet.cuda()
        acc = get_acc(ppnet, dataloader)
    else:
        acc = None

    data = {v: {
        "tau": [0.0],
        "min_num_weights": [v],
        "prototypes": [num_prototypes],
        "weights": [num_weights],
        "acc": [acc]
    } for v in min_weights_values}
    for tau, min_weights in combos:
        min_weights = int(min_weights)
        if dataset is not None:
            ppnet = torch.load(model_path)
            ppnet.cuda()
            weights_to_prune = Pruner.prune_low_weights(ppnet, tau, min_weights=min_weights)
            acc = get_acc(ppnet, dataloader)
            data[min_weights]["acc"].append(acc)
            acc_str = f", Acc: {acc*100:.2f}"
            weights_to_prune = weights_to_prune.detach().cpu().numpy()
        else:
            weights_to_prune = Pruner.get_weights_to_prune(ppnet, tau, min_weights=min_weights)
            acc_str = ""

        num_pruned = weights_to_prune.sum()
        low_weight_prototypes = np.all(weights_to_prune, axis=0)

        data[min_weights]["tau"].append(tau)
        data[min_weights]["min_num_weights"].append(min_weights)
        data[min_weights]["prototypes"].append(num_prototypes - low_weight_prototypes.sum())
        data[min_weights]["weights"].append(num_weights - num_pruned)
        print(f"Tau: {tau:.2f}, Min Weights/Logit: {min_weights}, P: {data[min_weights]['prototypes'][-1]}, W: {data[min_weights]['weights'][-1]}{acc_str}")

    fig, ax = plt.subplots()
    if dataset is not None:
        ax2 = ax.twinx()
    for min_weights in min_weights_values:
        ax.plot(data[min_weights]["tau"], data[min_weights]["prototypes"], label=min_weights)
        if dataset is not None:
            ax2.plot(data[min_weights]["tau"], data[min_weights]["acc"], linestyle="--", label=min_weights)
    plt.legend(frameon=False)
    plt.xlabel("Tau")
    ax.set_ylabel("No. prototypes")
    ax2.set_ylabel("Accuracy")
    ax2.set_xlabel("Tau")
    ax.set_xlabel("Tau")
    plt.tight_layout()
    plt.show()

    flattened_weights = weights.flatten()
    sns.displot(flattened_weights[flattened_weights > 0.05])
    plt.show()

    print("Done!")


def get_acc(ppnet, dataloader):
    dataset_length = len(dataloader.dataset)
    predictions = np.zeros(dataset_length)
    targets = np.array(dataloader.dataset.targets)

    i = 0
    for batch_idx, (image, label) in enumerate(dataloader):
        with torch.no_grad():
            output, min_distances, similarity_maps = ppnet(image.cuda())

        next_i = i + len(image)
        predictions[i:next_i] = output.cpu().numpy().argmax(axis=1)
        i += len(image)

    return (predictions == targets).sum() / dataset_length


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model", help="Path to trained ProtoPNet .pth model")
    parser.add_argument("--tau", nargs=2, type=float, default=(0.05, 2.5))
    parser.add_argument("--tau-step", type=float, default=0.05)
    parser.add_argument("--min-weights", nargs=2, type=int, default=(0, 3))
    parser.add_argument('--gpuid', nargs=1, type=str, default=None)
    parser.add_argument("--workers", type=int, default=4, help="No. workers for dataloading")
    parser.add_argument("--dataset", type=str, default=None, help="Dataset to (optionally) test on")
    args = parser.parse_args()

    main(args.model, args.tau, args.tau_step, args.min_weights, args.dataset, args.workers)
