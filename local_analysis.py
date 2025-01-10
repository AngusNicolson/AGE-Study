
from pathlib import Path
import torch
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image

import re

import os
import copy

from helpers import makedir, find_high_activation_crop
import model
import push
import train_and_test as tnt
import save
from log import Logger
from preprocess import mean, std, preprocess_input_function, undo_preprocess_input_function

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--gpuid', nargs=1, type=str, default='0')
parser.add_argument('--model', help="Path to model")
parser.add_argument('--img', help="Path to image. The local analysis will be performed in the parent directory.")
parser.add_argument('--imgclass', nargs=1, type=int, default=-1)
parser.add_argument("--test", action="store_true", help="Run model on test set and report results")
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpuid[0]

# specify the test image to be analyzed
test_image_path = Path(args.img)
test_image_dir = test_image_path.parent  # './local_analysis/Painted_Bunting_Class15_0081/'
test_image_name = test_image_path.name  # 'Painted_Bunting_0081_15230.jpg'
test_image_label = args.imgclass[0]  # 15

# load the model
check_test_accu = args.test

load_model_path = Path(args.model)
load_model_dir = load_model_path.parent  # './saved_models/vgg19/003/'
load_model_name = load_model_path.name  # '10_18push0.7822.pth'

model_base_architecture = load_model_dir.parent.stem
experiment_run = load_model_dir.name

save_analysis_path = test_image_dir / model_base_architecture / experiment_run / load_model_name
save_analysis_path.mkdir(exist_ok=True, parents=True)

log = Logger(log_filename=save_analysis_path / 'local_analysis.log')

epoch_number_str = re.search(r'\d+', load_model_name).group(0)
start_epoch_number = int(epoch_number_str)

log('load model from ' + str(load_model_path))
log('model base architecture: ' + model_base_architecture)
log('experiment run: ' + experiment_run)

ppnet = torch.load(load_model_path)
ppnet = ppnet.cuda()
ppnet_multi = torch.nn.DataParallel(ppnet)
ppnet_multi.eval()

img_size = ppnet_multi.module.img_size
prototype_shape = ppnet.prototype_shape
max_dist = prototype_shape[1] * prototype_shape[2] * prototype_shape[3]

class_specific = True

normalize = transforms.Normalize(mean=mean,
                                 std=std)

# load the test data and check test accuracy
from settings import test_dir
if check_test_accu:
    test_batch_size = 100

    test_dataset = datasets.ImageFolder(
        test_dir,
        transforms.Compose([
            transforms.Resize(size=(img_size, img_size)),
            transforms.ToTensor(),
            normalize,
        ]))
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=test_batch_size, shuffle=False,
        num_workers=4, pin_memory=False)
    log('test set size: {0}'.format(len(test_loader.dataset)))

    accu = tnt.test(model=ppnet_multi, dataloader=test_loader,
                    class_specific=class_specific, log=print)

##### SANITY CHECK
# confirm prototype class identity
load_img_dir = load_model_dir / 'img'

prototype_info = np.load(load_img_dir / ('epoch-'+epoch_number_str) / ('bb'+epoch_number_str+'.npy'))
prototype_img_identity = prototype_info[:, -1]

log('Prototypes are chosen from ' + str(len(set(prototype_img_identity))) + ' number of classes.')
log('Their class identities are: ' + str(prototype_img_identity))

# confirm prototype connects most strongly to its own class
prototype_max_connection = torch.argmax(ppnet.last_layer.weight, dim=0)
prototype_max_connection = prototype_max_connection.cpu().numpy()
if np.sum(prototype_max_connection == prototype_img_identity) == ppnet.num_prototypes:
    log('All prototypes connect most strongly to their respective classes.')
else:
    log('WARNING: Not all prototypes connect most strongly to their respective classes.')
    log("Max connections are to: " + str(prototype_max_connection))


##### HELPER FUNCTIONS FOR PLOTTING
def save_preprocessed_img(fname, preprocessed_imgs, index=0):
    img_copy = copy.deepcopy(preprocessed_imgs[index:index+1])
    undo_preprocessed_img = undo_preprocess_input_function(img_copy)
    print('image index {0} in batch'.format(index))
    undo_preprocessed_img = undo_preprocessed_img[0]
    undo_preprocessed_img = undo_preprocessed_img.detach().cpu().numpy()
    undo_preprocessed_img = np.transpose(undo_preprocessed_img, [1,2,0])
    
    plt.imsave(fname, undo_preprocessed_img)
    return undo_preprocessed_img


def save_prototype(fname, epoch, index):
    p_img = plt.imread(os.path.join(load_img_dir, 'epoch-'+str(epoch), 'prototype-img'+str(index)+'.png'))
    #plt.axis('off')
    plt.imsave(fname, p_img)


def save_prototype_self_activation(fname, epoch, index):
    p_img = plt.imread(os.path.join(load_img_dir, 'epoch-'+str(epoch),
                                    'prototype-img-original_with_self_act'+str(index)+'.png'))
    #plt.axis('off')
    plt.imsave(fname, p_img)


def save_prototype_original_img_with_bbox(fname, epoch, index,
                                          bbox_height_start, bbox_height_end,
                                          bbox_width_start, bbox_width_end, color=(0, 255, 255)):
    p_img_bgr = cv2.imread(os.path.join(load_img_dir, 'epoch-'+str(epoch), 'prototype-img-original'+str(index)+'.png'))
    cv2.rectangle(p_img_bgr, (bbox_width_start, bbox_height_start), (bbox_width_end-1, bbox_height_end-1),
                  color, thickness=2)
    p_img_rgb = p_img_bgr[...,::-1]
    p_img_rgb = np.float32(p_img_rgb) / 255
    #plt.imshow(p_img_rgb)
    #plt.axis('off')
    plt.imsave(fname, p_img_rgb)


def imsave_with_bbox(fname, img_rgb, bbox_height_start, bbox_height_end,
                     bbox_width_start, bbox_width_end, color=(0, 255, 255)):
    img_bgr_uint8 = cv2.cvtColor(np.uint8(255*img_rgb), cv2.COLOR_RGB2BGR)
    cv2.rectangle(img_bgr_uint8, (bbox_width_start, bbox_height_start), (bbox_width_end-1, bbox_height_end-1),
                  color, thickness=2)
    img_rgb_uint8 = img_bgr_uint8[...,::-1]
    img_rgb_float = np.float32(img_rgb_uint8) / 255
    #plt.imshow(img_rgb_float)
    #plt.axis('off')
    plt.imsave(fname, img_rgb_float)


def upscale_activation(activation):
    if type(activation) is torch.Tensor:
        activation = activation.detach().cpu().numpy()

    upscaled_activation = cv2.resize(activation, dsize=(img_size, img_size), interpolation=cv2.INTER_CUBIC)
    return upscaled_activation


def get_heatmap_from_activations(upscaled_activation, scale=True):
    activation = upscaled_activation - np.min(upscaled_activation)
    if scale:
        activation = activation / np.max(activation)

    heatmap = cv2.applyColorMap(np.uint8(255*activation), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    heatmap = heatmap[..., ::-1]
    return heatmap


def overlay_activations_on_img(img, upscaled_activation, scale=True):
    heatmap = get_heatmap_from_activations(upscaled_activation, scale)
    overlayed_img = 0.5 * img + 0.3 * heatmap
    return overlayed_img


def plot_all_prototype_activations(activation_patterns, num_classes, num_prototypes, img=None, savefig=None, scale=True, titles=None):
    num_prototypes = int(num_prototypes)
    num_classes = int(num_classes)
    fig, axes = plt.subplots(num_classes, num_prototypes, figsize=(num_prototypes, num_classes))
    for i, ax in enumerate(axes.flatten()):
        upsampled_activations = upscale_activation(activation_patterns[0, i])
        if img is None:
            img_plot = upsampled_activations
        else:
            img_plot = overlay_activations_on_img(img, upsampled_activations, scale)
        ax.imshow(img_plot)
        ax.axis("off")
        if titles is not None:
            ax.set_title(titles[i])
    plt.tight_layout()
    if savefig is None:
        plt.show()
    else:
        plt.savefig(savefig)


# load the test image and forward it through the network
preprocess = transforms.Compose([
   transforms.Resize((img_size, img_size)),
   transforms.ToTensor(),
   normalize
])

img_pil = Image.open(test_image_path)
img_pil = img_pil.convert("RGB")
img_tensor = preprocess(img_pil)
img_variable = Variable(img_tensor.unsqueeze(0))

images_test = img_variable.cuda()
labels_test = torch.tensor([test_image_label])

logits, min_distances, similarity_maps = ppnet_multi(images_test)
conv_output, distances = ppnet.push_forward(images_test)
prototype_activations = ppnet.distance_2_similarity(min_distances)
prototype_activation_patterns = ppnet.distance_2_similarity(distances)
if ppnet.prototype_activation_function == 'linear':
    prototype_activations = prototype_activations + max_dist
    prototype_activation_patterns = prototype_activation_patterns + max_dist

tables = []
for i in range(logits.size(0)):
    tables.append((torch.argmax(logits, dim=1)[i].item(), labels_test[i].item()))
    log(str(i) + ' ' + str(tables[-1]))

idx = 0
predicted_cls = tables[idx][0]
correct_cls = tables[idx][1]
log('Predicted: ' + str(predicted_cls))
log('Actual: ' + str(correct_cls))
original_img = save_preprocessed_img(os.path.join(save_analysis_path, 'original_img.png'),
                                     images_test, idx)

plot_all_prototype_activations(
    prototype_activation_patterns,
    ppnet.num_classes,
    ppnet.num_prototypes / ppnet.num_classes,
    img=original_img,
    savefig=save_analysis_path / "all_prototype_activations_overlayed.png"
)

plot_all_prototype_activations(
    prototype_activation_patterns,
    ppnet.num_classes,
    ppnet.num_prototypes / ppnet.num_classes,
    savefig=save_analysis_path / "all_prototype_activations.png"
)

prototype_order_idx = np.argsort(prototype_max_connection)
prototype_max_connection_ordered = np.sort(prototype_max_connection)
if np.isnan(ppnet.prototype_class_identity.detach().cpu().numpy()).all():
    plot_all_prototype_activations(
        prototype_activation_patterns[:, prototype_order_idx],
        ppnet.num_classes,
        ppnet.num_prototypes / ppnet.num_classes,
        img=original_img,
        titles=prototype_max_connection_ordered,
        savefig=save_analysis_path / "all_prototype_activations_overlayed_sorted.png"
    )

    plot_all_prototype_activations(
        prototype_activation_patterns[:, prototype_order_idx],
        ppnet.num_classes,
        ppnet.num_prototypes / ppnet.num_classes,
        titles=prototype_max_connection_ordered,
        savefig=save_analysis_path / "all_prototype_activations_sorted.png"
    )


##### MOST ACTIVATED (NEAREST) 10 PROTOTYPES OF THIS IMAGE
makedir(os.path.join(save_analysis_path, 'most_activated_prototypes'))

log('Most activated 10 prototypes of this image:')
array_act, sorted_indices_act = torch.sort(prototype_activations[idx])
for i in range(1, 11):
    log('top {0} activated prototype for this image:'.format(i))
    save_prototype(os.path.join(save_analysis_path, 'most_activated_prototypes',
                                'top-%d_activated_prototype.png' % i),
                   start_epoch_number, sorted_indices_act[-i].item())
    save_prototype_original_img_with_bbox(fname=os.path.join(save_analysis_path, 'most_activated_prototypes',
                                                             'top-%d_activated_prototype_in_original_pimg.png' % i),
                                          epoch=start_epoch_number,
                                          index=sorted_indices_act[-i].item(),
                                          bbox_height_start=prototype_info[sorted_indices_act[-i].item()][1],
                                          bbox_height_end=prototype_info[sorted_indices_act[-i].item()][2],
                                          bbox_width_start=prototype_info[sorted_indices_act[-i].item()][3],
                                          bbox_width_end=prototype_info[sorted_indices_act[-i].item()][4],
                                          color=(0, 255, 255))
    save_prototype_self_activation(os.path.join(save_analysis_path, 'most_activated_prototypes',
                                                'top-%d_activated_prototype_self_act.png' % i),
                                   start_epoch_number, sorted_indices_act[-i].item())
    log('prototype index: {0}'.format(sorted_indices_act[-i].item()))
    log('prototype class identity: {0}'.format(prototype_img_identity[sorted_indices_act[-i].item()]))
    if prototype_max_connection[sorted_indices_act[-i].item()] != prototype_img_identity[sorted_indices_act[-i].item()]:
        log('prototype connection identity: {0}'.format(prototype_max_connection[sorted_indices_act[-i].item()]))
    log('activation value (similarity score): {0}'.format(array_act[-i]))
    log('last layer connection with predicted class: {0}'.format(ppnet.last_layer.weight[predicted_cls][sorted_indices_act[-i].item()]))
    
    activation_pattern = prototype_activation_patterns[idx][sorted_indices_act[-i].item()].detach().cpu().numpy()
    upsampled_activation_pattern = upscale_activation(activation_pattern)
    
    # show the most highly activated patch of the image by this prototype
    high_act_patch_indices = find_high_activation_crop(upsampled_activation_pattern)
    high_act_patch = original_img[high_act_patch_indices[0]:high_act_patch_indices[1],
                                  high_act_patch_indices[2]:high_act_patch_indices[3], :]
    log('most highly activated patch of the chosen image by this prototype:')
    #plt.axis('off')
    plt.imsave(os.path.join(save_analysis_path, 'most_activated_prototypes',
                            'most_highly_activated_patch_by_top-%d_prototype.png' % i),
               high_act_patch)
    log('most highly activated patch by this prototype shown in the original image:')
    imsave_with_bbox(fname=os.path.join(save_analysis_path, 'most_activated_prototypes',
                            'most_highly_activated_patch_in_original_img_by_top-%d_prototype.png' % i),
                     img_rgb=original_img,
                     bbox_height_start=high_act_patch_indices[0],
                     bbox_height_end=high_act_patch_indices[1],
                     bbox_width_start=high_act_patch_indices[2],
                     bbox_width_end=high_act_patch_indices[3], color=(0, 255, 255))
    
    # show the image overlayed with prototype activation map
    overlayed_img = overlay_activations_on_img(original_img, upsampled_activation_pattern)
    log('prototype activation map of the chosen image:')
    #plt.axis('off')
    plt.imsave(os.path.join(save_analysis_path, 'most_activated_prototypes',
                            'prototype_activation_map_by_top-%d_prototype.png' % i),
               overlayed_img)
    log('--------------------------------------------------------------')

##### PROTOTYPES FROM TOP-k CLASSES
k = 10
log('Prototypes from top-%d classes:' % k)
topk_logits, topk_classes = torch.topk(logits[idx], k=k)
for i, c in enumerate(topk_classes.detach().cpu().numpy()):
    prototype_identities = ppnet.prototype_class_identity.detach().cpu().numpy()[:, c]
    if np.isnan(prototype_identities).all():
        print("Class agnostic model")
        out_dir = save_analysis_path / "prototype_activations"
    else:
        out_dir = save_analysis_path / ('top-%d_class_prototypes' % (i+1))
    out_dir.mkdir(exist_ok=True, parents=True)

    log('top %d predicted class: %d' % (i+1, c))
    log('logit of the class: %f' % topk_logits[i])
    class_prototype_indices = np.nonzero(prototype_identities)[0]
    class_prototype_activations = prototype_activations[idx][class_prototype_indices]
    _, sorted_indices_cls_act = torch.sort(class_prototype_activations)

    prototype_cnt = 1
    for j in reversed(sorted_indices_cls_act.detach().cpu().numpy()):
        prototype_index = class_prototype_indices[j]
        save_prototype(out_dir / ('top-%d_activated_prototype.png' % prototype_cnt),
                       start_epoch_number, prototype_index)
        save_prototype_original_img_with_bbox(fname=out_dir / ('top-%d_activated_prototype_in_original_pimg.png' % prototype_cnt),
                                              epoch=start_epoch_number,
                                              index=prototype_index,
                                              bbox_height_start=prototype_info[prototype_index][1],
                                              bbox_height_end=prototype_info[prototype_index][2],
                                              bbox_width_start=prototype_info[prototype_index][3],
                                              bbox_width_end=prototype_info[prototype_index][4],
                                              color=(0, 255, 255))
        save_prototype_self_activation(out_dir / ('top-%d_activated_prototype_self_act.png' % prototype_cnt),
                                       start_epoch_number, prototype_index)
        log('prototype index: {0}'.format(prototype_index))
        log('prototype class identity: {0}'.format(prototype_img_identity[prototype_index]))
        if prototype_max_connection[prototype_index] != prototype_img_identity[prototype_index]:
            log('prototype connection identity: {0}'.format(prototype_max_connection[prototype_index]))
        log('activation value (similarity score): {0}'.format(prototype_activations[idx][prototype_index]))
        log('last layer connection: {0}'.format(ppnet.last_layer.weight[c][prototype_index]))
        
        activation_pattern = prototype_activation_patterns[idx][prototype_index].detach().cpu().numpy()
        upsampled_activation_pattern = upscale_activation(activation_pattern)
        
        # show the most highly activated patch of the image by this prototype
        high_act_patch_indices = find_high_activation_crop(upsampled_activation_pattern)
        high_act_patch = original_img[high_act_patch_indices[0]:high_act_patch_indices[1],
                                      high_act_patch_indices[2]:high_act_patch_indices[3], :]
        log('most highly activated patch of the chosen image by this prototype:')
        plt.imsave(out_dir / ('most_highly_activated_patch_by_top-%d_prototype.png' % prototype_cnt),
                   high_act_patch)
        log('most highly activated patch by this prototype shown in the original image:')
        imsave_with_bbox(fname=out_dir / ('most_highly_activated_patch_in_original_img_by_top-%d_prototype.png' % prototype_cnt),
                         img_rgb=original_img,
                         bbox_height_start=high_act_patch_indices[0],
                         bbox_height_end=high_act_patch_indices[1],
                         bbox_width_start=high_act_patch_indices[2],
                         bbox_width_end=high_act_patch_indices[3], color=(0, 255, 255))
        
        # show the image overlayed with prototype activation map
        overlayed_img = overlay_activations_on_img(original_img, upsampled_activation_pattern)
        log('prototype activation map of the chosen image:')
        plt.imsave(out_dir / ('prototype_activation_map_by_top-%d_prototype.png' % prototype_cnt), overlayed_img)
        log('--------------------------------------------------------------')
        prototype_cnt += 1
    log('***************************************************************')
    if np.isnan(prototype_identities).all():
        break

if predicted_cls == correct_cls:
    log('Prediction is correct.')
else:
    log(f'Prediction is wrong. Pred: {predicted_cls}, GT: {correct_cls}')

log.close()

print("Done!")

