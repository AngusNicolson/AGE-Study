
import json
import os
import shutil
import time
from pathlib import Path
from argparse import ArgumentParser

import Augmentor
import matplotlib.pyplot as plt
import torch
import torch.utils.data
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from sklearn.metrics import confusion_matrix
from torch.nn.functional import cross_entropy
from torch.utils.tensorboard import SummaryWriter

from resnet_model import resnet18_wrapper
import save
from log import Logger
from plotting import plot_confusion_matrix
from preprocess import mean, std
from settings import train_dir, test_dir

resnet_models = {
    "resnet18": resnet18_wrapper
}


def main(json_path, workers=4):
    json_path = Path(json_path)
    with open(json_path, "r") as fp:
        params = json.load(fp)

    model = create_model(params["dropout"], params["dropout2d"], params["num_classes"], params["base_architecture"])
    model = model.cuda()
    trainer = Trainer(model, json_path, workers)
    trainer.train()
    trainer.log.close()
    print("Done!")


def create_model(dropout=0.5, dropout_2d=False, num_classes=10, base_architecture="resnet18", pretrained=True):
    model = resnet_models[base_architecture](pretrained=pretrained, dropout_2d=dropout_2d)
    model_size = model.fc.weight.shape[-1]
    # NB: Newly initialised layers have requires_grad=True
    model.fc = torch.nn.Sequential(
        torch.nn.Dropout(dropout),
        torch.nn.Linear(model_size, num_classes)
    )
    model.dropout = dropout
    model.pretrained = pretrained
    model.last_layer = model.fc[1]
    return model


class Trainer:
    def __init__(self, model, json_path: Path, workers: int = 4):
        with open(json_path, "r") as fp:
            params = json.load(fp)
        self.params = params
        self.workers = workers
        self.model = model

        self.dataloaders = self.setup_dataloaders()

        self.epoch = 0

        self.model_dir = self.setup_dirs(json_path)

        self.log = Logger(log_filename=self.model_dir / 'train.log')
        tensorboard_dir = Path(self.log.log_filename).parent / "tensorboard"
        self.writer = SummaryWriter(str(tensorboard_dir))

        self.log(f"Hyperparameters: {self.params}")
        self.log("")
        self.log(f'training set size: {len(self.dataloaders["train"].dataset)}')
        self.log(f'test set size: {len(self.dataloaders["test"].dataset)}')
        self.log(f'batch size: {self.params["train_batch_size"]}')
        self.log("")

        self.optimizer, self.lr_scheduler = self.create_optimizers()

    def get_model_dir(self):
        return Path('./saved_models/plain/' + self.params["base_architecture"] + '/' + self.params["experiment_run"] + '/')

    def setup_dataloaders(self):
        normalize = transforms.Normalize(mean=mean, std=std)
        data_augmentation = self.get_augs(self.params["aug_params"])

        # all datasets
        # train set
        train_dataset = datasets.ImageFolder(
            train_dir,
            data_augmentation
        )
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=self.params["train_batch_size"], shuffle=True,
            num_workers=self.workers, pin_memory=False)
        # test set
        test_dataset = datasets.ImageFolder(
            test_dir,
            transforms.Compose([
                transforms.Resize(size=(self.params["img_size"], self.params["img_size"])),
                transforms.ToTensor(),
                normalize,
            ]))
        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=self.params["test_batch_size"], shuffle=False,
            num_workers=self.workers, pin_memory=False)

        dataloaders = {"train": train_loader, "test": test_loader}
        return dataloaders

    @staticmethod
    def get_augs(aug_params):
        normalize = transforms.Normalize(mean=mean, std=std)

        p = Augmentor.Pipeline()
        p.skew(**aug_params["skew"])
        p.shear(**aug_params["shear"])
        p.random_distortion(**aug_params["distortion"])

        augs = transforms.Compose([
            p.torch_transform(),
            transforms.ToTensor(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(**aug_params["rotation"]),
            transforms.ColorJitter(**aug_params["color_jitter"]),
            transforms.RandomResizedCrop(**aug_params["crop"]),
            transforms.Lambda(lambda x: x + torch.randn_like(x) * aug_params["noise"]["magnitude"]),  # Gaussian noise
            normalize
        ])
        return augs

    @staticmethod
    def accuracy(output, target, topk=(1,)):
        """Computes the precision@k for the specified values of k"""
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].float().sum()
            acc = correct_k / batch_size
            res.append(acc * 100)
        return res

    def test(self, tensorboard=True):
        dataloader = self.dataloaders["test"]
        self.log('\ttest')
        self.model.eval()
        return self._train_or_test_loop(dataloader=dataloader, optimizer=None, tensorboard=tensorboard)

    def _train_or_test_loop(self, dataloader, optimizer=None, tensorboard=True, frozen_weights=None):
        is_train = optimizer is not None
        # Assumes 20 iterations per last layer optimisation
        start = time.time()
        n_examples = 0
        n_correct = 0
        n_batches = 0
        total_cross_entropy = 0
        total_loss = 0

        logits = []
        labels = []
        for i, (image, label) in enumerate(dataloader):
            input = image.cuda()
            target = label.cuda()

            # torch.enable_grad() has no effect outside of no_grad()
            grad_req = torch.enable_grad() if is_train else torch.no_grad()
            with grad_req:
                # nn.Module has implemented __call__() function
                # so no need to call .forward
                output = self.model(input)

                # compute loss
                losses = self.loss_fn(output, target)

                # evaluation statistics
                _, predicted = torch.max(output.data, 1)
                n_examples += target.size(0)
                n_correct += (predicted == target).sum().item()

                logits.append(output.detach().cpu())
                labels.append(target.detach().cpu())

                n_batches += 1
                total_cross_entropy += losses["cross_entropy"].item()
                acc = (target.detach() == predicted.detach()).float().mean().item()
                mae = (target.detach() - predicted.detach()).float().abs().mean().item()

            total_loss += losses["total"].item()

            # compute gradient and do SGD step
            if is_train:
                optimizer.zero_grad()
                losses["total"].backward()
                optimizer.step()
                if tensorboard:
                    n_iterations = n_examples + self.epoch * len(dataloader.dataset)
                    self.writer.add_scalar(f"iteration/Loss",
                                           losses["total"].item(), n_iterations)
                    self.writer.add_scalar(f"iteration/CELoss",
                                           losses["cross_entropy"].item(), n_iterations)
                    self.writer.add_scalar(f"iteration/Acc",
                                           acc, n_iterations)
                    self.writer.add_scalar(f"iteration/ClassMAE",
                                           mae, n_iterations)
                    self.writer.add_histogram(f"iteration/last_layer_weights",
                                              self.model.last_layer.weight.data.detach().cpu(), n_iterations)

            del input
            del target
            del output
            del predicted
            del losses

        end = time.time()

        logits = torch.concat(logits)
        labels = torch.concat(labels)
        topk = {k: 0 for k in (1, 2, 3)}
        topk[1], topk[2], topk[3] = self.accuracy(logits, labels, (1, 2, 3))

        preds = logits.argmax(dim=1)
        cm = confusion_matrix(y_true=labels, y_pred=preds)
        cm_figure = plot_confusion_matrix(cm)

        class_mae = (labels - preds).abs().float().mean().item()

        self.log('\ttime: \t{0}'.format(end - start))
        self.log(f"\tLoss: \t{total_loss / n_batches}")
        self.log('\tcross ent: \t{0}'.format(total_cross_entropy / n_batches))
        for i in (1, 2, 3):
            self.log(f'\ttop{i}:\t{topk[i]} %')
        self.log(f"\tClass MAE:\t{class_mae}")

        writer_type = "train" if is_train else "val"
        if tensorboard:
            for i in (1, 2, 3):
                self.writer.add_scalar(f"Acc/Top-{i}/{writer_type}", topk[i], self.epoch)
            self.writer.add_scalar(f"Acc/ClassMAE/{writer_type}", class_mae, self.epoch)
            self.writer.add_scalar(f"Loss/Total/{writer_type}", total_loss / n_batches, self.epoch)
            self.writer.add_scalar(f"Loss/CrossEntropy/{writer_type}", total_cross_entropy / n_batches, self.epoch)

            if is_train:
                self.writer.add_histogram(f"Epoch/last_layer_weights",
                                          self.model.last_layer.weight.data.detach().cpu(), self.epoch)
                self.writer.add_scalar(f"Loss/lr", optimizer.param_groups[0]["lr"], self.epoch)

            plt.tight_layout()
            self.writer.add_figure(f"Confusion Matrix /{writer_type}", cm_figure, self.epoch)

        return n_correct / n_examples

    def loss_fn(self, output, target):
        # compute loss
        cross_entropy_loss = cross_entropy(output, target)
        loss = cross_entropy_loss

        losses = {
            "cross_entropy": cross_entropy_loss,
            "total": loss
        }
        return losses

    def train(self):
        self.log('start training')
        for epoch in range(self.params["num_train_epochs"]):
            self.log('epoch: \t{0}'.format(epoch))
            _ = self.train_loop()
            self.lr_scheduler.step()
            accu = self.test()
            save.save_model_w_condition(model=self.model, model_dir=self.model_dir, model_name=str(epoch) + 'nopush',
                                        accu=accu, target_accu=self.params["target_accuracy"], log=self.log)

    def train_loop(self):
        optimizer = self.optimizer
        dataloader = self.dataloaders["train"]
        self.epoch += 1

        self.log('\ttrain')
        self.model.train()
        return self._train_or_test_loop(dataloader=dataloader, optimizer=optimizer)

    def create_optimizers(self):
        optimizer_specs = \
            {'params': self.model.parameters(), 'lr': self.params["lr"], 'weight_decay': self.params["weight_decay"]},
        optimizer = torch.optim.Adam(optimizer_specs)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.params["lr_step_size"], gamma=0.1)

        return optimizer, lr_scheduler

    def setup_dirs(self, json_path: Path):
        model_dir = self.get_model_dir()
        model_dir.mkdir(exist_ok=True, parents=True)
        cwd = Path(os.getcwd())
        shutil.copy(src=cwd / __file__, dst=model_dir)
        shutil.copy(src=cwd / 'settings.py', dst=model_dir)
        shutil.copy(src=cwd / "resnet_model.py", dst=model_dir)
        if json_path.parent != model_dir:
            shutil.copy(src=json_path, dst=model_dir)

        return model_dir


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--gpuid', nargs=1, type=str, default=None)  # python3 main.py -gpuid=0,1,2,3
    parser.add_argument("--json", type=str, default="./plain_hyperparameters.json",
                        help="A .json containing the hyperparameters for the experiment to run.")
    parser.add_argument("--workers", type=int, default=4, help="No. workers for dataloading")
    args = parser.parse_args()

    if args.gpuid is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpuid[0]
        print(os.environ['CUDA_VISIBLE_DEVICES'])

    main(args.json, args.workers)
