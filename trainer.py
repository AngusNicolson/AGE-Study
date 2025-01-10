
import json
import os
import re
import shutil
import time
from pathlib import Path
from collections import Counter

import Augmentor
import matplotlib.pyplot as plt
import torch
import torch.utils.data
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from sklearn.metrics import confusion_matrix
from torch.nn.functional import cross_entropy
from torch.utils.tensorboard import SummaryWriter
import numpy as np

import push
import save
from helpers import list_of_distances
from log import Logger
from loss import CoherenceLoss, NumProtoLoss
from plotting import plot_confusion_matrix
from preprocess import mean, std, preprocess_input_function
from settings import train_dir, test_dir, train_push_dir
import find_nearest


class BaseTrainer:
    def __init__(self, ppnet, ppnet_multi, json_path: Path, workers: int = 4):
        with open(json_path, "r") as fp:
            params = json.load(fp)
        self.params = params
        self.workers = workers
        self.ppnet = ppnet
        self.ppnet_multi = ppnet_multi

        self.class_specific = self.params["class_specific"]
        self.use_l1_mask = self.params["use_l1_mask"]
        self.coefs = self.params["coefs"]
        self.coherence_loss_fn = CoherenceLoss((self.ppnet.proto_layer_rf_info[0], self.ppnet.proto_layer_rf_info[0]))
        self.num_proto_loss_fn = NumProtoLoss(self.params["num_proto_loss"], reduce=True)
        self.dataloaders = self.setup_dataloaders()

        self.epoch = 0
        self.last_only_epochs = 0

        # Must define in main class
        self.log = None
        self.writer = None

    def get_model_dir(self):
        if self.params["val_fold"] is None:
            fold_str = ""
        else:
            fold_str = f'fold-{self.params["val_fold"]}/'
        return Path(f'./saved_models/{self.params["base_architecture"]}/{self.params["experiment_run"]}/{fold_str}')

    def setup_dataloaders(self):
        normalize = transforms.Normalize(mean=mean, std=std)

        data_augmentation = self.get_augs(self.params["aug_params"])

        push_transforms = transforms.Compose([
                    transforms.Resize(size=(self.params["img_size"], self.params["img_size"])),
                    transforms.ToTensor(),
        ])

        test_transforms = transforms.Compose([
                    transforms.Resize(size=(self.params["img_size"], self.params["img_size"])),
                    transforms.ToTensor(),
                    normalize,
        ])

        # If not cross_val just use directories in defined in settings
        # Otherwise, assume train_dir and train_push_dir directories in settings contain split folds
        # and use all for training apart from the specified "val_fold"
        if self.params["val_fold"] is None:
            train_dataset = datasets.ImageFolder(train_dir, data_augmentation)
            train_push_dataset = datasets.ImageFolder(train_push_dir, push_transforms)
            test_dataset = datasets.ImageFolder(test_dir, test_transforms)
        else:
            folds = [d for d in Path(train_dir).glob("*") if d.is_dir()]
            folds.sort()
            test_fold = folds[self.params["val_fold"]]
            train_folds = [d for d in folds if d != test_fold]

            train_datasets = [datasets.ImageFolder(d, data_augmentation) for d in train_folds]
            train_dataset = torch.utils.data.ConcatDataset(train_datasets)

            train_push_dirs = [Path(train_push_dir) / d.name for d in train_folds]
            train_push_datasets = [datasets.ImageFolder(d, push_transforms) for d in train_push_dirs]
            train_push_dataset = torch.utils.data.ConcatDataset(train_push_datasets)

            test_dataset = datasets.ImageFolder(Path(train_push_dir) / test_fold.name, test_transforms)

        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=self.params["train_batch_size"], shuffle=True,
            num_workers=self.workers, pin_memory=False)

        train_push_loader = torch.utils.data.DataLoader(
            train_push_dataset, batch_size=self.params["train_push_batch_size"], shuffle=False,
            num_workers=self.workers, pin_memory=False)

        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=self.params["test_batch_size"], shuffle=False,
            num_workers=self.workers, pin_memory=False)

        dataloaders = {"train": train_loader, "train_push": train_push_loader, "test": test_loader}
        return dataloaders

    @staticmethod
    def get_augs(aug_params):

        def noise(magnitude=0.0):
            """Additive gaussian noise"""
            return transforms.Lambda(lambda x: x + torch.randn_like(x) * magnitude)

        aug_funcs = {
            "horizontal_flip": transforms.RandomHorizontalFlip,
            "rotation": transforms.RandomRotation,
            "color_jitter": transforms.ColorJitter,
            "crop": transforms.RandomResizedCrop,
            "resize": transforms.Resize,
            "noise": noise
        }

        normalize = transforms.Normalize(mean=mean, std=std)

        p = Augmentor.Pipeline()
        if "skew" in aug_params.keys():
            p.skew(**aug_params["skew"])
        if "shear" in aug_params.keys():
            p.shear(**aug_params["shear"])
        if "distortion" in aug_params.keys():
            p.random_distortion(**aug_params["distortion"])

        if len(p.operations) != 0:
            aug_list = [p.torch_transform()]
        else:
            aug_list = []

        aug_list = aug_list + [transforms.ToTensor()]
        for name, aug_func in aug_funcs.items():
            if name in aug_params.keys():
                aug_list = aug_list + [aug_func(**aug_params[name])]

        aug_list.append(normalize)

        augs = transforms.Compose(aug_list)
        return augs

    def last_only(self):
        for p in self.ppnet_multi.module.features.parameters():
            p.requires_grad = False
        for p in self.ppnet_multi.module.add_on_layers.parameters():
            p.requires_grad = False
        self.ppnet_multi.module.prototype_vectors.requires_grad = False
        for p in self.ppnet_multi.module.last_layer.parameters():
            p.requires_grad = True

        self.log('\tlast layer')

    def warm_only(self):
        for p in self.ppnet_multi.module.features.parameters():
            p.requires_grad = False
        for p in self.ppnet_multi.module.add_on_layers.parameters():
            p.requires_grad = True
        self.ppnet_multi.module.prototype_vectors.requires_grad = True
        for p in self.ppnet_multi.module.last_layer.parameters():
            p.requires_grad = True

        self.log('\twarm')

    def joint(self):
        for p in self.ppnet_multi.module.features.parameters():
            p.requires_grad = True
        for p in self.ppnet_multi.module.add_on_layers.parameters():
            p.requires_grad = True
        self.ppnet_multi.module.prototype_vectors.requires_grad = True
        for p in self.ppnet_multi.module.last_layer.parameters():
            p.requires_grad = True

        self.log('\tjoint')

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

    def test(self, last_only=False, tensorboard=True):
        dataloader = self.dataloaders["test"]
        self.log('\ttest')
        self.ppnet_multi.eval()
        return self._train_or_test_loop(dataloader=dataloader, optimizer=None, last_only=last_only, tensorboard=tensorboard)

    def _train_or_test_loop(self, dataloader, optimizer=None, last_only=False, tensorboard=True, frozen_weights=None):
        is_train = optimizer is not None
        last_str = f"LastOnly" if last_only else ""
        # Assumes 20 iterations per last layer optimisation
        counter = self.last_only_epochs if last_only else self.epoch
        start = time.time()
        n_examples = 0
        n_correct = 0
        n_batches = 0
        total_cross_entropy = 0
        total_coherence_loss = 0
        total_num_proto_loss = 0
        total_num_proto_by_class = torch.zeros(self.ppnet.num_classes)
        total_coherence_loss_by_part = torch.zeros(self.ppnet.prototype_shape[0])
        total_cluster_cost = 0
        # separation cost is meaningful only for class_specific
        total_separation_cost = 0
        total_avg_separation_cost = 0
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
                output, min_distances, similarity_maps = self.ppnet_multi(input)

                # compute loss
                losses = self.loss_fn(output, min_distances, similarity_maps, target, label)

                # evaluation statistics
                _, predicted = torch.max(output.data, 1)
                n_examples += target.size(0)
                n_correct += (predicted == target).sum().item()

                logits.append(output.detach().cpu())
                labels.append(target.detach().cpu())

                n_batches += 1
                total_cross_entropy += losses["cross_entropy"].item()
                total_coherence_loss += losses["coherence"].item()
                total_num_proto_loss += losses["num_proto"].item()
                total_coherence_loss_by_part += losses["coherence_by_part"].detach().cpu()
                total_num_proto_by_class += losses["num_proto_by_class"].detach().cpu()
                total_cluster_cost += losses["cluster_cost"].item()
                total_separation_cost += losses["separation_cost"].item()
                total_avg_separation_cost += losses["avg_separation_cost"].item()
                acc = (target.detach() == predicted.detach()).float().mean().item()
                mae = (target.detach() - predicted.detach()).float().abs().mean().item()

            total_loss += losses["total"].item()

            # compute gradient and do SGD step
            if is_train:
                optimizer.zero_grad()
                losses["total"].backward()
                if frozen_weights is not None:
                    # Copy old weights into new to remove changes
                    # Cannot simply set gradient to 0 for frozen weights as momentum could still change them
                    frozen_last_layer_weights = self.ppnet.last_layer.weight.data[frozen_weights]
                    optimizer.step()
                    self.ppnet.last_layer.weight.data[frozen_weights] = frozen_last_layer_weights
                    del frozen_last_layer_weights
                else:
                    optimizer.step()
                if tensorboard:
                    n_iterations = n_examples + counter * len(dataloader.dataset)
                    self.writer.add_scalar(f"iteration{last_str}/Loss",
                                           losses["total"].item(), n_iterations)
                    self.writer.add_scalar(f"iteration{last_str}/CELoss",
                                           losses["cross_entropy"].item(), n_iterations)
                    self.writer.add_scalar(f"iteration{last_str}/CoherenceLoss",
                                           losses["coherence"].item(), n_iterations)
                    self.writer.add_scalar(f"iteration{last_str}/NumProtoLoss",
                                           losses["num_proto"].item(), n_iterations)
                    self.writer.add_histogram(f"iteration{last_str}/NumProto",
                                           losses["num_proto_by_class"].detach().cpu(), n_iterations)
                    self.writer.add_scalar(f"iteration{last_str}/ClusterLoss",
                                           losses["cluster_cost"].item(), n_iterations)
                    self.writer.add_scalar(f"iteration{last_str}/SepLoss",
                                           losses["separation_cost"].item(), n_iterations)
                    self.writer.add_scalar(f"iteration{last_str}/l1Loss",
                                           losses["l1"].item(), n_iterations)
                    self.writer.add_scalar(f"iteration{last_str}/Acc",
                                           acc, n_iterations)
                    self.writer.add_scalar(f"iteration{last_str}/ClassMAE",
                                           mae, n_iterations)
                    self.writer.add_histogram(f"iteration/last_layer_weights",
                                              self.ppnet.last_layer.weight.data.detach().cpu(), n_iterations)
                    if not last_only:
                        self.writer.add_histogram(f"iteration/Coherence",
                                                  losses["coherence_by_part"].detach().cpu(), n_iterations)
                    else:
                        self.writer.add_scalar(f"iteration{last_str}/l1LossMasked",
                                               losses["l1_masked"].item(), n_iterations)
                        self.writer.add_scalar(f"iteration{last_str}/l1LossAll",
                                               losses["l1_all"].item(), n_iterations)

            del input
            del target
            del output
            del predicted
            del min_distances
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
        self.log('\tcoherence: \t{0}'.format(total_coherence_loss / n_batches))
        self.log('\tnum_proto: \t{0}'.format(total_num_proto_loss / n_batches))
        self.log('\tcluster: \t{0}'.format(total_cluster_cost / n_batches))
        if self.class_specific:
            self.log('\tseparation:\t{0}'.format(total_separation_cost / n_batches))
            self.log('\tavg separation:\t{0}'.format(total_avg_separation_cost / n_batches))
        for i in (1, 2, 3):
            self.log(f'\ttop{i}:\t{topk[i]} %')
        self.log(f"\tClass MAE:\t{class_mae}")
        self.log('\tl1: \t\t{0}'.format(self.ppnet_multi.module.last_layer.weight.norm(p=1).item()))
        p = self.ppnet_multi.module.prototype_vectors.view(self.ppnet_multi.module.num_prototypes, -1).cpu()
        with torch.no_grad():
            p_avg_pair_dist = torch.mean(list_of_distances(p, p))
        self.log('\tp dist pair: \t{0}'.format(p_avg_pair_dist.item()))

        writer_type = "train" if is_train else "val"
        if tensorboard:
            for i in (1, 2, 3):
                self.writer.add_scalar(f"Acc{last_str}/Top-{i}/{writer_type}", topk[i], counter)
            self.writer.add_scalar(f"Acc{last_str}/ClassMAE/{writer_type}", class_mae, counter)
            self.writer.add_scalar(f"Loss{last_str}/Total/{writer_type}", total_loss / n_batches, counter)
            self.writer.add_scalar(f"Loss{last_str}/CrossEntropy/{writer_type}", total_cross_entropy / n_batches, counter)
            self.writer.add_scalar(f"Loss{last_str}/Coherence/{writer_type}", total_coherence_loss / n_batches, counter)
            self.writer.add_scalar(f"Loss{last_str}/NumProto/{writer_type}", total_num_proto_loss / n_batches, counter)
            self.writer.add_scalar(f"Loss{last_str}/Cluster/{writer_type}", total_cluster_cost / n_batches, counter)
            self.writer.add_scalar(f"Loss{last_str}/Separation/{writer_type}", total_separation_cost / n_batches, counter)
            self.writer.add_scalar(f"Loss{last_str}/AvgSeparation/{writer_type}", total_avg_separation_cost / n_batches, counter)
            self.writer.add_scalar(f"Loss{last_str}/L1/{writer_type}", self.ppnet_multi.module.last_layer.weight.norm(p=1).item(), counter)

            if is_train and last_only:
                self.writer.add_histogram(f"Epoch/last_layer_weights",
                                          self.ppnet.last_layer.weight.data.detach().cpu(), counter)

            if not last_only:
                self.writer.add_scalar(f"Loss/AvgPairDist/{writer_type}", p_avg_pair_dist.item(), counter)
                self.writer.add_histogram(f"Epoch/Coherence/{writer_type}",
                                          total_coherence_loss_by_part / n_batches, counter)
                if is_train:
                    self.writer.add_scalar(f"Loss/lr", optimizer.param_groups[0]["lr"], counter)
            plt.tight_layout()
            if self.params["do_confusion_plot"]:
                self.writer.add_figure(f"Confusion Matrix {last_str}/{writer_type}", cm_figure, counter)

        return n_correct / n_examples

    def loss_fn(self, output, min_distances, similarity_maps, target, label):
        # compute loss
        cross_entropy_loss = cross_entropy(output, target)

        coherence_loss_by_part = self.coherence_loss_fn(similarity_maps).mean(dim=0)
        coherence_loss = coherence_loss_by_part.mean()

        similarities = self.ppnet.distance_2_similarity(min_distances)
        contributions = similarities.unsqueeze(-1) * self.ppnet.last_layer.weight.T
        num_proto_loss_by_class = self.num_proto_loss_fn(contributions.abs()).mean(dim=0)
        num_proto_loss = num_proto_loss_by_class.mean()

        if self.class_specific:
            max_dist = (self.ppnet_multi.module.prototype_shape[1]
                        * self.ppnet_multi.module.prototype_shape[2]
                        * self.ppnet_multi.module.prototype_shape[3])

            # prototypes_of_correct_class is a tensor of shape batch_size * num_prototypes
            # calculate cluster cost
            prototypes_of_correct_class = torch.t(self.ppnet_multi.module.prototype_class_identity[:, label]).cuda()
            inverted_distances, _ = torch.max((max_dist - min_distances) * prototypes_of_correct_class, dim=1)
            cluster_cost = torch.mean(max_dist - inverted_distances)

            # calculate separation cost
            prototypes_of_wrong_class = 1 - prototypes_of_correct_class
            inverted_distances_to_nontarget_prototypes, _ = \
                torch.max((max_dist - min_distances) * prototypes_of_wrong_class, dim=1)
            separation_cost = torch.mean(max_dist - inverted_distances_to_nontarget_prototypes)

            # calculate avg separation cost
            avg_separation_cost = \
                torch.sum(min_distances * prototypes_of_wrong_class, dim=1) / torch.sum(
                    prototypes_of_wrong_class, dim=1)
            avg_separation_cost = torch.mean(avg_separation_cost)

            l1_all = self.ppnet_multi.module.last_layer.weight.norm(p=1)
            l1_mask = 1 - torch.t(self.ppnet_multi.module.prototype_class_identity).cuda()
            l1_masked = (self.ppnet_multi.module.last_layer.weight * l1_mask).norm(p=1)

            if self.use_l1_mask:
                l1 = l1_masked
            else:
                l1 = l1_all

            loss = (self.coefs['crs_ent'] * cross_entropy_loss
                    + self.coefs['clst'] * cluster_cost
                    + self.coefs['sep'] * separation_cost
                    + self.coefs['l1'] * l1
                    + self.coefs["coherence"] * coherence_loss
                    + self.coefs["num_proto"] * num_proto_loss
                    )
        else:
            min_distance, _ = torch.min(min_distances, dim=1)
            cluster_cost = torch.mean(min_distance)
            l1_all = self.ppnet_multi.module.last_layer.weight.norm(p=1)
            l1 = l1_all
            separation_cost = torch.tensor([torch.nan])
            avg_separation_cost = torch.tensor([torch.nan])
            l1_masked = torch.tensor([torch.nan])

            loss = (self.coefs['crs_ent'] * cross_entropy_loss
                    + self.coefs['clst'] * cluster_cost
                    + self.coefs['l1'] * l1
                    + self.coefs["coherence"] * coherence_loss
                    + self.coefs["num_proto"] * num_proto_loss
                    )

        losses = {
            "cross_entropy": cross_entropy_loss,
            "coherence": coherence_loss,
            "coherence_by_part": coherence_loss_by_part,
            "num_proto": num_proto_loss,
            "num_proto_by_class": num_proto_loss_by_class,
            "cluster_cost": cluster_cost,
            "separation_cost": separation_cost,
            "avg_separation_cost": avg_separation_cost,
            "l1": l1,
            "l1_masked": l1_masked,
            "l1_all": l1_all,
            "total": loss
        }
        return losses


class Trainer(BaseTrainer):
    def __init__(self, ppnet, ppnet_multi, json_path: Path, workers: int = 4):
        super(Trainer, self).__init__(ppnet, ppnet_multi, json_path, workers)

        self.model_dir, self.img_dir = self.setup_dirs(json_path)

        self.log = Logger(log_filename=self.model_dir / 'train.log')
        tensorboard_dir = Path(self.log.log_filename).parent / "tensorboard"
        self.writer = SummaryWriter(str(tensorboard_dir))

        self.log(f"Hyperparameters: {self.params}")
        self.log("")
        self.log(f"Train path: {train_dir}")
        self.log(f'training set size: {len(self.dataloaders["train"].dataset)}')
        self.log(f"Push path: {train_push_dir}")
        self.log(f'push set size: {len(self.dataloaders["train_push"].dataset)}')
        if self.params["val_fold"] is None:
            self.log(f"Test path: {test_dir}")
        else:
            self.log(f'Val fold: {self.params["val_fold"]}')
        self.log(f'test set size: {len(self.dataloaders["test"].dataset)}')
        self.log(f'batch size: {self.params["train_batch_size"]}')
        self.log("")

        # define optimizer
        self.optimizers, self.joint_lr_scheduler = self.create_optimizers()

    def train(self):
        weight_matrix_filename = 'outputL_weights'
        prototype_img_filename_prefix = 'prototype-img'
        prototype_self_act_filename_prefix = 'prototype-self-act'
        proto_bound_boxes_filename_prefix = 'bb'

        push_epochs = list(range(
            self.params["push_start"],
            self.params["num_train_epochs"],
            self.params["push_epochs_frequency"]
        ))

        self.log('start training')
        for epoch in range(self.params["num_train_epochs"]):
            self.log('epoch: \t{0}'.format(epoch))

            if epoch < self.params["num_warm_epochs"]:
                _ = self.train_loop("warm")
            else:
                _ = self.train_loop("joint")

            accu = self.test()
            save.save_model_w_condition(model=self.ppnet, model_dir=self.model_dir, model_name=str(epoch) + 'nopush', accu=accu,
                                        target_accu=self.params["target_accuracy"], log=self.log)

            if epoch >= self.params["push_start"] and epoch in push_epochs:
                push.push_prototypes(
                    self.dataloaders["train_push"],  # pytorch dataloader (must be unnormalized in [0,1])
                    prototype_network_parallel=self.ppnet_multi,  # pytorch network with prototype_vectors
                    class_specific=self.class_specific,
                    preprocess_input_function=preprocess_input_function,  # normalize if needed
                    prototype_layer_stride=1,
                    root_dir_for_saving_prototypes=self.img_dir,  # if not None, prototypes will be saved here
                    epoch_number=epoch,  # if not provided, prototypes saved previously will be overwritten
                    prototype_img_filename_prefix=prototype_img_filename_prefix,
                    prototype_self_act_filename_prefix=prototype_self_act_filename_prefix,
                    proto_bound_boxes_filename_prefix=proto_bound_boxes_filename_prefix,
                    save_prototype_class_identity=True,
                    log=self.log)
                # Don't overwrite tensorboard metrics for current epoch after push
                accu = self.test(tensorboard=False)
                save.save_model_w_condition(model=self.ppnet, model_dir=self.model_dir, model_name=str(epoch) + 'push', accu=accu,
                                            target_accu=self.params["target_accuracy"], log=self.log)

                for i in range(self.params["num_last_only_iter"]):
                    self.log(f'iteration: \t{i}')
                    _ = self.train_loop("last_only")
                    accu = self.test(last_only=True)
                    save.save_model_w_condition(model=self.ppnet, model_dir=self.model_dir,
                                                model_name=str(epoch) + '_' + str(i) + 'push', accu=accu,
                                                target_accu=self.params["target_accuracy"], log=self.log)

    def train_loop(self, train_type):
        dataloader = None
        optimizer = None
        last_only = False
        if train_type == "warm":
            self.warm_only()
            optimizer = self.optimizers["warm"]
            dataloader = self.dataloaders["train"]
            self.epoch += 1
        elif train_type == "joint":
            self.joint()
            self.joint_lr_scheduler.step()
            dataloader = self.dataloaders["train"]
            optimizer = self.optimizers["joint"]
            self.epoch += 1
        elif train_type == "last_only":
            if self.params["prototype_activation_function"] != 'linear':
                self.last_only()
                dataloader = self.dataloaders["train"]
                optimizer = self.optimizers["last_layer"]
                self.last_only_epochs += 1
            last_only = True
        else:
            raise NotImplementedError(f"train_type {train_type} not implemented!")

        self.log('\ttrain')
        self.ppnet_multi.train()
        return self._train_or_test_loop(dataloader=dataloader, optimizer=optimizer, last_only=last_only)

    def create_optimizers(self):
        joint_optimizer_specs = \
            [{'params': self.ppnet.features.parameters(), 'lr': self.params["joint_optimizer_lrs"]['features'],
              'weight_decay': self.params["weight_decay"]["joint"]["features"]},
             # bias are now also being regularized
             {'params': self.ppnet.add_on_layers.parameters(), 'lr': self.params["joint_optimizer_lrs"]['add_on_layers'],
              'weight_decay': self.params["weight_decay"]["joint"]["add_on_layers"]},
             {'params': self.ppnet.prototype_vectors, 'lr': self.params["joint_optimizer_lrs"]['prototype_vectors']},
             ]
        joint_optimizer = torch.optim.Adam(joint_optimizer_specs)
        joint_lr_scheduler = torch.optim.lr_scheduler.StepLR(joint_optimizer, step_size=self.params["joint_lr_step_size"],
                                                             gamma=0.1)

        warm_optimizer_specs = \
            [{'params': self.ppnet.add_on_layers.parameters(), 'lr': self.params["warm_optimizer_lrs"]['add_on_layers'],
              'weight_decay': self.params["weight_decay"]["warm"]["add_on_layers"]},
             {'params': self.ppnet.prototype_vectors, 'lr': self.params["warm_optimizer_lrs"]['prototype_vectors']},
             ]
        warm_optimizer = torch.optim.Adam(warm_optimizer_specs)

        last_layer_optimizer_specs = [{'params': self.ppnet.last_layer.parameters(), 'lr': self.params["last_layer_optimizer_lr"],
                                       "weight_decay": self.params["weight_decay"]["last_layer"]}]
        last_layer_optimizer = torch.optim.Adam(last_layer_optimizer_specs)

        optimizers = {"joint": joint_optimizer, "warm": warm_optimizer, "last_layer": last_layer_optimizer}
        return optimizers, joint_lr_scheduler

    def setup_dirs(self, json_path: Path):
        base_architecture_type = re.match('^[a-z]*', self.params["base_architecture"]).group(0)
        model_dir = self.get_model_dir()
        model_dir.mkdir(exist_ok=True, parents=True)
        cwd = Path(os.getcwd())
        shutil.copy(src=cwd / __file__, dst=model_dir)
        shutil.copy(src=cwd / 'settings.py', dst=model_dir)
        if json_path.parent != model_dir:
            shutil.copy(src=json_path, dst=model_dir)
        shutil.copy(src=cwd / f"{base_architecture_type}_features.py", dst=model_dir)
        shutil.copy(src=cwd / 'model.py', dst=model_dir)
        shutil.copy(src=cwd / 'trainer.py', dst=model_dir)

        img_dir = model_dir / 'img'
        img_dir.mkdir(exist_ok=True, parents=True)
        return model_dir, img_dir


class Pruner(BaseTrainer):
    def __init__(self, orig_model_path: Path, json_path: Path, workers: int = 4):
        # Load trained ProtoPNet model
        ppnet = torch.load(orig_model_path)
        ppnet = ppnet.cuda()
        ppnet_multi = torch.nn.DataParallel(ppnet)
        orig_json_path = orig_model_path.parent / "hyperparameters.json"
        super(Pruner, self).__init__(ppnet, ppnet_multi, orig_json_path, workers)

        self.orig_model_path = orig_model_path
        self.orig_model_dir = self.orig_model_path.parent

        # Load hyperparameters for pruning
        with open(json_path, "r") as fp:
            self.prune_params = json.load(fp)

        self.coefs = self.prune_params["coefs"]
        self.params["target_accuracy"] = self.prune_params["target_accuracy"]

        self.model_dir, self.orig_epoch = self.setup_dirs()

        self.log = Logger(log_filename=self.model_dir / 'prune.log')
        tensorboard_dir = self.model_dir / "tensorboard"
        self.writer = SummaryWriter(str(tensorboard_dir))

        self.log(f"Hyperparameters: {self.params}")
        self.log("")
        self.log(f"Prune Hyperparameters {self.prune_params}")
        self.log("")

        # define optimizer
        self.optimizers = self.create_optimizers()
        self.frozen_weights = None

    def setup_dirs(self):
        orig_model_name = self.orig_model_path.name

        need_push = ('nopush' in orig_model_name)
        if need_push:
            raise ValueError("Model not pushed. Pruning must happen after a push.")  # pruning must happen after push
        else:
            epoch = orig_model_name.split('push')[0]

        if '_' in epoch:
            epoch = int(epoch.split('_')[0])
        else:
            epoch = int(epoch)

        model_dir = self.orig_model_dir / f'pruned_prototypes_' \
                                          f'epoch{epoch}_' \
                                          f'k{self.prune_params["k"]}_' \
                                          f'ct{self.prune_params["class_threshold"]}_' \
                                          f'wt{self.prune_params["weight_threshold"]:.4f}_' \
                                          f'{self.prune_params["suffix"]}'
        model_dir.mkdir(exist_ok=True)
        shutil.copy(src=os.path.join(os.getcwd(), __file__), dst=model_dir)
        shutil.copy(src=os.path.join(os.getcwd(), "settings.py"), dst=model_dir)
        return model_dir, epoch

    def create_optimizers(self):
        last_layer_optimizer_specs = [{'params': self.ppnet.last_layer.parameters(), 'lr': self.prune_params["lr"],
                                       "weight_decay": self.prune_params["weight_decay"]}]
        last_layer_optimizer = torch.optim.Adam(last_layer_optimizer_specs)

        optimizers = {"last_layer": last_layer_optimizer}
        return optimizers

    def prune(self, optimize_last_layer=True, copy_prototype_imgs=True):
        multi_label_prototypes = []
        low_weight_prototypes = []
        if self.prune_params["weight_threshold"] > 0:
            low_weight_prototypes = self.get_low_weight_prototypes()
        if self.prune_params["class_threshold"] > 0:
            multi_label_prototypes = self.get_multi_label_prototypes()

        prototypes_to_prune = list(set(low_weight_prototypes).union(set(multi_label_prototypes)))
        self.log(f"{len(prototypes_to_prune)} total prototypes to prune.")

        prune_info = self.create_prune_info(prototypes_to_prune)
        np.save(str(self.model_dir / 'prune_info.npy'), prune_info)

        if copy_prototype_imgs:
            self.copy_prototype_imgs(prototypes_to_prune)

        self.ppnet.prune_prototypes(prototypes_to_prune)

        # Determine frozen weights AFTER pruning the prototypes
        self.frozen_weights = self.prune_low_weights_()
        if not self.prune_params["freeze_weights"]:
            self.frozen_weights = None

        self.test(last_only=True)
        self.last_only_epochs += 1

        if optimize_last_layer:
            self.train()

    def get_weights_to_prune_(self):
        weights = self.get_weights_to_prune(
            self.ppnet,
            self.prune_params["weight_threshold"],
            self.prune_params["min_weights"]
        )
        return weights

    @staticmethod
    def get_weights_to_prune(ppnet, weight_threshold, min_weights=0):
        weights = ppnet.last_layer.weight.detach().cpu().numpy()
        less_than_threshold = np.abs(weights) < weight_threshold
        for class_idx in range(ppnet.num_classes):
            if (~less_than_threshold[class_idx]).sum() < min_weights:
                # Order negative to get the largest weights first
                sorted_idx = np.argsort(-np.abs(weights[class_idx]))
                for i in range(min_weights):
                    less_than_threshold[class_idx][sorted_idx[i]] = False

        return less_than_threshold

    @staticmethod
    def prune_low_weights(ppnet, weight_threshold, min_weights=0):
        less_than_threshold = Pruner.get_weights_to_prune(ppnet, weight_threshold, min_weights)

        less_than_threshold = torch.tensor(less_than_threshold)
        ppnet.last_layer.weight.data[less_than_threshold] = 0.0
        return less_than_threshold

    def prune_low_weights_(self):
        less_than_threshold = self.prune_low_weights(
            self.ppnet,
            self.prune_params['weight_threshold'],
            self.prune_params['min_weights']
        )
        num_weights = self.ppnet.last_layer.weight.detach().cpu().numpy().size

        self.log(f"Setting weights lower than {self.prune_params['weight_threshold']} to 0")
        self.log(f"{less_than_threshold.sum()}/{num_weights} weights lower than threshold")
        return less_than_threshold

    def get_low_weight_prototypes(self):
        less_than_threshold = self.get_weights_to_prune_()
        low_weight_prototypes = np.all(less_than_threshold, axis=0)
        prototypes_to_prune = np.where(low_weight_prototypes)[0]

        self.log(f"weight_threshold = {self.prune_params['weight_threshold']}")
        self.log(f"{len(prototypes_to_prune)} prototypes will be pruned")

        return prototypes_to_prune

    def get_multi_label_prototypes(self):
        nearest_train_patch_class_ids = find_nearest.find_k_nearest_patches_to_prototypes(
            dataloader=self.dataloaders["train_push"],
            prototype_network_parallel=self.ppnet_multi,
            k=self.prune_params["k"],
            preprocess_input_function=preprocess_input_function,
            full_save=False,
            log=self.log
        )
        original_num_prototypes = self.ppnet.num_prototypes

        prototypes_to_prune = []
        for j in range(original_num_prototypes):
            class_j = torch.argmax(self.ppnet.prototype_class_identity[j]).item()
            nearest_train_patch_class_counts_j = Counter(nearest_train_patch_class_ids[j])
            # if no such element is in Counter, it will return 0
            if nearest_train_patch_class_counts_j[class_j] < self.prune_params["class_threshold"]:
                prototypes_to_prune.append(j)

        self.log('k = {}, class_threshold = {}'.format(self.prune_params["k"], self.prune_params["class_threshold"]))
        self.log('{} prototypes will be pruned'.format(len(prototypes_to_prune)))

        return prototypes_to_prune

    def create_prune_info(self, prototypes_to_prune):
        """
        :param prototypes_to_prune: List of prototype indices
        :return: np.array of prototype indices to prune and their associated class
        """
        class_of_prototypes_to_prune = torch.argmax(
            self.ppnet.prototype_class_identity[prototypes_to_prune],
            dim=1
        ).numpy().reshape(-1, 1)
        prototypes_to_prune_np = np.array(prototypes_to_prune).reshape(-1, 1)
        prune_info = np.hstack((prototypes_to_prune_np, class_of_prototypes_to_prune))
        return prune_info

    def copy_prototype_imgs(self, prototypes_to_prune):
        original_num_prototypes = self.ppnet.num_prototypes
        original_img_dir = self.orig_model_dir / 'img' / f'epoch-{self.orig_epoch}'
        dst_img_dir = self.model_dir / 'img' / f'epoch-{self.orig_epoch}'
        dst_img_dir.mkdir(exist_ok=True, parents=True)

        prototypes_to_keep = list(set(range(original_num_prototypes)) - set(prototypes_to_prune))

        img_strings = ["prototype-img", "prototype-img-original", "prototype-img-original_with_self_act"]

        for idx in range(len(prototypes_to_keep)):
            for img_str in img_strings:
                shutil.copyfile(
                    src=original_img_dir / f'{img_str}{prototypes_to_keep[idx]}.png',
                    dst=dst_img_dir / f'{img_str}{idx}.png'
                )

            shutil.copyfile(
                src=original_img_dir / f"prototype-self-act{prototypes_to_keep[idx]}.npy",
                dst=dst_img_dir / f"prototype-self-act{idx}.npy"
            )

            bb = np.load(str(original_img_dir / f"bb{self.orig_epoch}.npy"))
            bb = bb[prototypes_to_keep]
            np.save(str(dst_img_dir / f"bb{self.orig_epoch}.npy"), bb)

            bb_rf = np.load(str(original_img_dir / f"bb-receptive_field{self.orig_epoch}.npy"))
            bb_rf = bb_rf[prototypes_to_keep]
            np.save(str(dst_img_dir / f"bb-receptive_field{self.orig_epoch}.npy"), bb_rf)

    def train(self):
        self.last_only()
        self.log('start training')
        for epoch in range(self.prune_params["num_train_epochs"]):
            self.log('epoch: \t{0}'.format(epoch))
            _ = self._train_or_test_loop(
                dataloader=self.dataloaders["train"],
                optimizer=self.optimizers["last_layer"],
                last_only=True,
                frozen_weights=self.frozen_weights
            )
            accu = self.test(last_only=True, tensorboard=True)
            model_name = f"{self.orig_model_path.stem}_{self.last_only_epochs}_prune"
            save.save_model_w_condition(model=self.ppnet, model_dir=self.model_dir, model_name=model_name,
                                        accu=accu,
                                        target_accu=self.params["target_accuracy"], log=self.log)
            self.last_only_epochs += 1
