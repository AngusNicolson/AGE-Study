
import time

import matplotlib.pyplot as plt
import torch
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import confusion_matrix

from helpers import list_of_distances, make_one_hot
from log import Logger
from plotting import plot_confusion_matrix


def _train_or_test(model, dataloader, optimizer=None, class_specific=True, use_l1_mask=True,
                   coefs=None, log: Logger = Logger("./default_log.txt"), epoch=None, writer: SummaryWriter = None,
                   last_only_iter: int = None, push_idx: int = None):
    """
    model: the multi-gpu model
    dataloader:
    optimizer: if None, will be test evaluation
    """
    if (last_only_iter is None) != (push_idx is None):
        raise ValueError(f"Either both or neither of last_only_iter and push_idx must be provided. "
                         f"last_only_iter: {last_only_iter}, push_idx: {push_idx}")

    is_train = optimizer is not None
    last_str = f"LastOnly" if last_only_iter is not None else ""
    # Assumes 20 iterations per last layer optimisation
    counter = last_only_iter + push_idx*20 if last_only_iter is not None else epoch
    start = time.time()
    n_examples = 0
    n_correct = 0
    n_batches = 0
    total_cross_entropy = 0
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
            output, min_distances, similarity_maps = model(input)

            # compute loss
            cross_entropy = torch.nn.functional.cross_entropy(output, target)

            if class_specific:
                max_dist = (model.module.prototype_shape[1]
                            * model.module.prototype_shape[2]
                            * model.module.prototype_shape[3])

                # prototypes_of_correct_class is a tensor of shape batch_size * num_prototypes
                # calculate cluster cost
                prototypes_of_correct_class = torch.t(model.module.prototype_class_identity[:, label]).cuda()
                inverted_distances, _ = torch.max((max_dist - min_distances) * prototypes_of_correct_class, dim=1)
                cluster_cost = torch.mean(max_dist - inverted_distances)

                # calculate separation cost
                prototypes_of_wrong_class = 1 - prototypes_of_correct_class
                inverted_distances_to_nontarget_prototypes, _ = \
                    torch.max((max_dist - min_distances) * prototypes_of_wrong_class, dim=1)
                separation_cost = torch.mean(max_dist - inverted_distances_to_nontarget_prototypes)

                # calculate avg separation cost
                avg_separation_cost = \
                    torch.sum(min_distances * prototypes_of_wrong_class, dim=1) / torch.sum(prototypes_of_wrong_class, dim=1)
                avg_separation_cost = torch.mean(avg_separation_cost)
                
                if use_l1_mask:
                    l1_mask = 1 - torch.t(model.module.prototype_class_identity).cuda()
                    l1 = (model.module.last_layer.weight * l1_mask).norm(p=1)
                else:
                    l1 = model.module.last_layer.weight.norm(p=1) 

            else:
                min_distance, _ = torch.min(min_distances, dim=1)
                cluster_cost = torch.mean(min_distance)
                l1 = model.module.last_layer.weight.norm(p=1)

            # evaluation statistics
            _, predicted = torch.max(output.data, 1)
            n_examples += target.size(0)
            n_correct += (predicted == target).sum().item()

            logits.append(output.detach().cpu())
            labels.append(target.detach().cpu())

            n_batches += 1
            total_cross_entropy += cross_entropy.item()
            total_cluster_cost += cluster_cost.item()
            total_separation_cost += separation_cost.item()
            total_avg_separation_cost += avg_separation_cost.item()
            acc = (target.detach() == predicted.detach()).float().mean().item()
            mae = (target.detach() - predicted.detach()).float().abs().mean().item()

        if class_specific:
            if coefs is not None:
                loss = (coefs['crs_ent'] * cross_entropy
                        + coefs['clst'] * cluster_cost
                        + coefs['sep'] * separation_cost
                        + coefs['l1'] * l1)
            else:
                loss = cross_entropy + 0.8 * cluster_cost - 0.08 * separation_cost + 1e-4 * l1
        else:
            if coefs is not None:
                loss = (coefs['crs_ent'] * cross_entropy
                        + coefs['clst'] * cluster_cost
                        + coefs['l1'] * l1)
            else:
                loss = cross_entropy + 0.8 * cluster_cost + 1e-4 * l1

        total_loss += loss.item()

        # compute gradient and do SGD step
        if is_train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if writer is not None and counter is not None:
                writer.add_scalar(f"iteration{last_str}/Loss", loss.item(), n_examples + counter*len(dataloader.dataset))
                writer.add_scalar(f"iteration{last_str}/CELoss", cross_entropy.item(), n_examples + counter*len(dataloader.dataset))
                writer.add_scalar(f"iteration{last_str}/ClusterLoss", cluster_cost.item(), n_examples + counter*len(dataloader.dataset))
                writer.add_scalar(f"iteration{last_str}/SepLoss", separation_cost.item(), n_examples + counter*len(dataloader.dataset))
                writer.add_scalar(f"iteration{last_str}/l1Loss", l1.item(), n_examples + counter*len(dataloader.dataset))
                writer.add_scalar(f"iteration{last_str}/Acc", acc, n_examples + counter * len(dataloader.dataset))
                writer.add_scalar(f"iteration{last_str}/ClassMAE", mae, n_examples + counter * len(dataloader.dataset))

        del input
        del target
        del output
        del predicted
        del min_distances

    end = time.time()

    logits = torch.concat(logits)
    labels = torch.concat(labels)
    topk = {k: 0 for k in (1, 2, 3)}
    topk[1], topk[2], topk[3] = accuracy(logits, labels, (1, 2, 3))

    preds = logits.argmax(dim=1)
    cm = confusion_matrix(y_true=labels, y_pred=preds)
    cm_figure = plot_confusion_matrix(cm)
    if writer is None:
        plt.tight_layout()
        plt.show()

    class_mae = (labels - preds).abs().float().mean().item()

    log('\ttime: \t{0}'.format(end - start))
    log(f"\tLoss: \t{total_loss / n_batches}")
    log('\tcross ent: \t{0}'.format(total_cross_entropy / n_batches))
    log('\tcluster: \t{0}'.format(total_cluster_cost / n_batches))
    if class_specific:
        log('\tseparation:\t{0}'.format(total_separation_cost / n_batches))
        log('\tavg separation:\t{0}'.format(total_avg_separation_cost / n_batches))
    for i in (1, 2, 3):
        log(f'\ttop{i}:\t{topk[i]} %')
    log(f"\tClass MAE:\t{class_mae}")
    log('\tl1: \t\t{0}'.format(model.module.last_layer.weight.norm(p=1).item()))
    p = model.module.prototype_vectors.view(model.module.num_prototypes, -1).cpu()
    with torch.no_grad():
        p_avg_pair_dist = torch.mean(list_of_distances(p, p))
    log('\tp dist pair: \t{0}'.format(p_avg_pair_dist.item()))

    if writer is not None and counter is not None:
        writer_type = "train" if is_train else "val"
        for i in (1, 2, 3):
            writer.add_scalar(f"Acc{last_str}/Top-{i}/{writer_type}", topk[i], counter)
        writer.add_scalar(f"Acc{last_str}/ClassMAE/{writer_type}", class_mae, counter)
        writer.add_scalar(f"Loss{last_str}/Total/{writer_type}", total_loss / n_batches, counter)
        writer.add_scalar(f"Loss{last_str}/CrossEntropy/{writer_type}", total_cross_entropy / n_batches, counter)
        writer.add_scalar(f"Loss{last_str}/Cluster/{writer_type}", total_cluster_cost / n_batches, counter)
        writer.add_scalar(f"Loss{last_str}/Separation/{writer_type}", total_separation_cost / n_batches, counter)
        writer.add_scalar(f"Loss{last_str}/AvgSeparation/{writer_type}", total_avg_separation_cost / n_batches, counter)
        writer.add_scalar(f"Loss{last_str}/L1/{writer_type}", model.module.last_layer.weight.norm(p=1).item(), counter)
        if last_only_iter is None:
            writer.add_scalar(f"Loss/AvgPairDist/{writer_type}", p_avg_pair_dist.item(), counter)
            if is_train:
                writer.add_scalar(f"Loss/lr", optimizer.param_groups[0]["lr"], counter)
        plt.tight_layout()
        writer.add_figure(f"Confusion Matrix {last_str}/{writer_type}", cm_figure, counter)

    return n_correct / n_examples


def train(model, dataloader, optimizer, class_specific=False, coefs=None, log=Logger("./default_log.txt"), epoch=None, writer=None, last_only_iter=None, push_idx=None):
    assert(optimizer is not None)
    
    log('\ttrain')
    model.train()
    return _train_or_test(model=model, dataloader=dataloader, optimizer=optimizer,
                          class_specific=class_specific, coefs=coefs, log=log, epoch=epoch, writer=writer, last_only_iter=last_only_iter, push_idx=push_idx)


def test(model, dataloader, class_specific=False, log=Logger("./default_log.txt"), epoch=None, writer=None, last_only_iter=None, push_idx=None):
    log('\ttest')
    model.eval()
    return _train_or_test(model=model, dataloader=dataloader, optimizer=None,
                          class_specific=class_specific, log=log, epoch=epoch, writer=writer, last_only_iter=last_only_iter, push_idx=push_idx)


def last_only(model, log=print):
    for p in model.module.features.parameters():
        p.requires_grad = False
    for p in model.module.add_on_layers.parameters():
        p.requires_grad = False
    model.module.prototype_vectors.requires_grad = False
    for p in model.module.last_layer.parameters():
        p.requires_grad = True
    
    log('\tlast layer')


def warm_only(model, log=print):
    for p in model.module.features.parameters():
        p.requires_grad = False
    for p in model.module.add_on_layers.parameters():
        p.requires_grad = True
    model.module.prototype_vectors.requires_grad = True
    for p in model.module.last_layer.parameters():
        p.requires_grad = True
    
    log('\twarm')


def joint(model, log=print):
    for p in model.module.features.parameters():
        p.requires_grad = True
    for p in model.module.add_on_layers.parameters():
        p.requires_grad = True
    model.module.prototype_vectors.requires_grad = True
    for p in model.module.last_layer.parameters():
        p.requires_grad = True
    
    log('\tjoint')


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
