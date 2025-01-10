
import os
import argparse
from pathlib import Path
import json


import torch
from model import construct_PPNet

from trainer import Trainer


def main(json_path, workers=4):
    json_path = Path(json_path)
    with open(json_path, "r") as fp:
        params = json.load(fp)

    ppnet, ppnet_multi = create_model(params)
    trainer = Trainer(ppnet, ppnet_multi, json_path, workers)
    trainer.train()
    trainer.log.close()
    print("Done!")


def create_model(params):
    ppnet = construct_PPNet(base_architecture=params["base_architecture"],
                            pretrained=True,
                            pretrained_path=params["pretrained_path"],
                            img_size=params["img_size"],
                            prototype_shape=tuple(params["prototype_shape"]),
                            num_classes=params["num_classes"],
                            prototype_activation_function=params["prototype_activation_function"],
                            add_on_layers_type=params["add_on_layers_type"],
                            dropout=params["dropout"],
                            class_specific=params["class_specific"])
    # if prototype_activation_function == 'linear':
    #    ppnet.set_last_layer_incorrect_connection(incorrect_strength=0)
    ppnet = ppnet.cuda()
    ppnet_multi = torch.nn.DataParallel(ppnet)
    return ppnet, ppnet_multi


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpuid', nargs=1, type=str, default=None)  # python3 main.py -gpuid=0,1,2,3
    parser.add_argument("--json", type=str, default="./hyperparameters.json",
                        help="A .json containing the hyperparameters for the experiment to run.")
    parser.add_argument("--workers", type=int, default=4, help="No. workers for dataloading")
    args = parser.parse_args()

    if args.gpuid is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpuid[0]
        print(os.environ['CUDA_VISIBLE_DEVICES'])

    main(args.json, args.workers)

