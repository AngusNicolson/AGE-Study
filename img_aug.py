import Augmentor
import os
def makedir(path):
    '''
    if path does not exist in the file system, create it
    '''
    if not os.path.exists(path):
        os.makedirs(path)

datasets_root_dir = '/home/lina3782/labs/protopnet/intergrowth/datasets/'
dir = datasets_root_dir + 'train/'
target_dir = datasets_root_dir + 'train_augmented/'

makedir(target_dir)
folders = [os.path.join(dir, folder) for folder in next(os.walk(dir))[1]]
target_folders = [os.path.join(target_dir, folder) for folder in next(os.walk(dir))[1]]

for i in range(len(folders)):
    fd = folders[i]
    tfd = target_folders[i]
    # rotation
    p = Augmentor.Pipeline(source_directory=fd, output_directory=tfd)
    p.rotate(probability=0.7, max_left_rotation=15, max_right_rotation=15)
    p.flip_left_right(probability=0.5)
    p.skew(probability=0.7, magnitude=0.2)
    p.shear(probability=0.7, max_shear_left=10, max_shear_right=10)
    p.crop_random(probability=1, percentage_area=0.8)
    p.resize(probability=1.0, width=224, height=224)
    for i in range(5):
        p.process()
    del p
