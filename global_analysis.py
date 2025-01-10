import torch
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import numpy as np
import cv2
import matplotlib.pyplot as plt

import re
from pathlib import Path
import os

from helpers import makedir
import find_nearest

from preprocess import preprocess_input_function

import argparse


def main(args):
    workers = args.workers
    load_model_path = Path(args.model)
    load_model_dir = load_model_path.parent
    load_model_name = load_model_path.name

    epoch_number_str = re.search(r'\d+', load_model_name).group(0)
    start_epoch_number = int(epoch_number_str)

    # load the model
    print('load model from ' + str(load_model_path))
    # torch.multiprocessing.set_sharing_strategy('file_system')
    ppnet = torch.load(load_model_path)
    ppnet = ppnet.cuda()
    # ppnet_multi = torch.nn.DataParallel(ppnet)

    img_size = ppnet.img_size

    # load the data
    # must use unaugmented (original) dataset
    from settings import train_push_dir, test_dir
    train_dir = train_push_dir

    batch_size = args.batch_size

    # do not normalize data
    train_loader = create_dataloader(train_dir, img_size=img_size, batch_size=batch_size, workers=workers)
    test_loader = create_dataloader(test_dir, img_size=img_size, batch_size=batch_size, workers=workers)

    root_dir_for_saving_train_images = load_model_dir / (load_model_name.split('.pth')[0] + '_nearest_train')
    root_dir_for_saving_test_images = load_model_dir / (load_model_name.split('.pth')[0] + '_nearest_test')
    makedir(root_dir_for_saving_train_images)
    makedir(root_dir_for_saving_test_images)

    # save prototypes in original images
    load_img_dir = os.path.join(load_model_dir, 'img')
    prototype_info = np.load(os.path.join(load_img_dir, 'epoch-'+str(start_epoch_number), 'bb'+str(start_epoch_number)+'.npy'))

    print("Saving prototypes...")
    for j in range(ppnet.num_prototypes):
        train_proto_dir = root_dir_for_saving_train_images / str(j)
        test_proto_dir = root_dir_for_saving_test_images / str(j)

        for d in [train_proto_dir, test_proto_dir]:
            d.mkdir(exist_ok=True, parents=True)
            save_prototype_original_img_with_bbox(fname=str(d / 'prototype_in_original_pimg.png'),
                                                  load_img_dir=load_img_dir,
                                                  epoch=start_epoch_number,
                                                  index=j,
                                                  bbox_height_start=prototype_info[j][1],
                                                  bbox_height_end=prototype_info[j][2],
                                                  bbox_width_start=prototype_info[j][3],
                                                  bbox_width_end=prototype_info[j][4],
                                                  color=(0, 255, 255))

    k = args.k
    find_patches_settings = [
        (test_loader, root_dir_for_saving_test_images, k),
        (train_loader, root_dir_for_saving_train_images, k + 1)
    ]
    for loader, save_dir, num_imgs in find_patches_settings:
        find_nearest.find_k_nearest_patches_to_prototypes(
            dataloader=loader,  # pytorch dataloader (must be unnormalized in [0,1])
            prototype_network_parallel=None,  # pytorch network with prototype_vectors
            prototype_network=ppnet,
            k=num_imgs,
            preprocess_input_function=preprocess_input_function,  # normalize if needed
            full_save=True,
            root_dir_for_saving_images=str(save_dir),
            log=print)


def create_dataloader(path: str, img_size: int = 224, batch_size: int = 64, shuffle: bool = True, workers: int = 4):
    dataset = datasets.ImageFolder(
        path,
        transforms.Compose([
            transforms.Resize(size=(img_size, img_size)),
            transforms.ToTensor(),
        ]))
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle,
        num_workers=workers, pin_memory=False)
    return loader


def save_prototype_original_img_with_bbox(fname, load_img_dir, epoch, index,
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


if __name__ == "__main__":
    # Usage: python3 global_analysis.py -modeldir='./saved_models/' -model=''
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpuid', help="Which GPU to use", nargs=1, type=str, default=None)
    parser.add_argument('--model', help="Path to the model (.pth)")
    parser.add_argument("--batch-size", help="Batch size for dataloaders", default=256, type=int)
    parser.add_argument("--k", help="Number of k nearest patches to prototypes to save.", default=5, type=int)
    parser.add_argument("--workers", help="No. workers for dataloading", default=4, type=int)
    args = parser.parse_args()
    if args.gpuid is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpuid[0]
    main(args)
