
import os
from argparse import ArgumentParser
from pathlib import Path

from trainer import Pruner


def main(json_path, workers, model_path):
    json_path = Path(json_path)
    model_path = Path(model_path)
    pruner = Pruner(model_path, json_path, workers=workers)
    pruner.prune(True)
    print("Done!")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--gpuid', nargs=1, type=str, default=None)
    parser.add_argument("--json", type=str, default="./prune_hyperparameters.json",
                        help="A .json containing the hyperparameters for the experiment to run.")
    parser.add_argument("--model", help="Path to trained ProtoPNet .pth model")
    parser.add_argument("--workers", type=int, default=4, help="No. workers for dataloading")
    args = parser.parse_args()

    if args.gpuid is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpuid[0]
        print(os.environ['CUDA_VISIBLE_DEVICES'])

    main(args.json, args.workers, args.model)
