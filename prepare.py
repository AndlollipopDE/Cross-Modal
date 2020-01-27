import os
import shutil
from scipy.io import loadmat
import numpy as np


def fileprepare(dir_path, if_train=False):
    name = None
    cam_dirs = []
    for i in range(6):
        cam_dirs.append(os.path.join(dir_path, 'cam' + str(i + 1)))
    if if_train:
        name = 'exp/train_id.mat'
        if not os.path.exists(os.path.join(dir_path, 'train')):
            os.mkdir(os.path.join(dir_path, 'train'))
    mat_path = os.path.join(dir_path, name)
    ids = loadmat(mat_path)['id'][0]
    if if_train:
        mat_path = os.path.join(dir_path, 'exp/val_id.mat')
        temp = loadmat(mat_path)['id'][0]
        ids = np.append(ids, temp)
    if if_train:
        if not os.path.exists(os.path.join(dir_path, 'train/rgb')):
            os.mkdir(os.path.join(dir_path, 'train/rgb'))
        if not os.path.exists(os.path.join(dir_path, 'train/ir')):
            os.mkdir(os.path.join(dir_path, 'train/ir'))
    for idx in ids:
        id_name = str(idx).zfill(4)
        for index, cam_dir in enumerate(cam_dirs):
            if index == 2 or index == 5:
                id_path_to = os.path.join(dir_path, 'train/ir/' + id_name)
            else:
                id_path_to = os.path.join(dir_path, 'train/rgb/' + id_name)
            if not os.path.exists(id_path_to):
                os.mkdir(id_path_to)
            id_path_from = os.path.join(cam_dir, id_name)
            if not os.path.exists(id_path_from):
                continue
            filenames = os.listdir(id_path_from)
            for filename in filenames:
                shutil.copyfile(os.path.join(id_path_from, filename), os.path.join(
                    id_path_to, 'cam'+str(index + 1)+'_'+filename))


fileprepare('./SYSU-MM01', True)
