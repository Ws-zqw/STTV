import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import os
import random

from torchvision.transforms.functional import rotate

class LimitedFoV(object):
    def __init__(self, fov=360.):
        self.fov = fov

    def __call__(self, x):
        # print(x.shape)
        angle = random.randint(0, 359)
        rotate_index = int(angle / 360. * x.shape[2])
        fov_index = int(self.fov / 360. * x.shape[2])
        if rotate_index > 0:
            img_shift = torch.zeros(x.shape)
            img_shift[:, :, :rotate_index] = x[:, :, -rotate_index:]
            img_shift[:, :, rotate_index:] = x[:, :, :(x.shape[2] - rotate_index)]
        else:
            img_shift = x
        return img_shift[:, :, :fov_index]

# STTV and STTVCF
class ShiftImage(object):
    def __init__(self, shift=1):
        self.shift = shift

    def __call__(self, x):
        _, _, width = x.shape
        shift_amount = int(width * self.shift / 360.)
        shifted_image = torch.cat((x[:, :, -shift_amount:], x[:, :, :-shift_amount]), dim=2)
        return shifted_image


def fill_padding(image_array, rotated_image_array):
    h, w = image_array.shape[-2], image_array.shape[-1]
    mask = torch.zeros_like(rotated_image_array)
    mask[rotated_image_array == 0] = 1
    avg = torch.zeros_like(rotated_image_array)
    avg[mask == 1] = image_array[mask == 1]
    avg_r, avg_g, avg_b = avg[0, :, :], avg[1, :, :], avg[2, :, :]
    avg_r, avg_g, avg_b = avg_r[avg_r != 0].to(torch.float32).mean(), avg_g[avg_g != 0].to(torch.float32).mean(), \
                          avg_b[avg_b != 0].to(torch.float32).mean()
    avg = torch.Tensor([avg_r, avg_g, avg_b]).reshape(3, 1, 1) * torch.ones((1, h, w))
    filled_image_array = rotated_image_array
    filled_image_array[mask == 1] = avg[mask == 1]
    return filled_image_array

class RotateImage(object):
    def __init__(self, angle=1):
        self.angle = angle

    def __call__(self, x):
        rotated_image = rotate(x, angle=-self.angle)
        # filled_image = fill_padding(x, rotated_image)
        # return filled_image
        return rotated_image

def input_transform_fov(size, fov):
    return transforms.Compose([
        transforms.Resize(size=tuple(size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
        LimitedFoV(fov=fov),
    ])


def input_transform(size):
    return transforms.Compose([
        transforms.Resize(size=tuple(size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


def input_transform_shift(size, shift):
    return transforms.Compose([
        transforms.Resize(size=tuple(size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
        ShiftImage(shift=shift),
    ])


def input_transform_rotate(size, angle):
    return transforms.Compose([
        transforms.Resize(size=tuple(size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
        RotateImage(angle=angle),
    ])

# pytorch version of CVUSA loader
class CVUSA(torch.utils.data.Dataset):
    def __init__(self, mode='', root='/data/zhangqingwang/dataset/CVUSA19/', same_area=True, print_bool=False,
                 args=None):  # CV-dataset
        super(CVUSA, self).__init__()

        self.args = args
        self.root = root
        self.mode = mode
        self.sat_size = [256, 256]
        # self.sat_size = [384, 384]
        self.sat_size_default = [256, 256]
        self.grd_size = [128, 512]
        # self.grd_size = [140, 768]

        if print_bool:
            print(self.sat_size, self.grd_size)

        self.sat_ori_size = [750, 750]
        self.grd_ori_size = [224, 1232]

        if args.fov != 0:
            self.transform_query = input_transform_fov(size=self.grd_size, fov=args.fov)
        else:
            self.transform_query = input_transform(size=self.grd_size)

        self.transform_reference = input_transform(size=self.sat_size)

        # Data Aug
        self.transform_shift = input_transform_shift
        self.transform_rotate = input_transform_rotate
        # self.transform_flip = input_transform_flip

        self.to_tensor = transforms.ToTensor()

        self.train_list = self.root + 'splits/train-19zl.csv'
        self.test_list = self.root + 'splits/val-19zl.csv'

        if print_bool:
            print('CVUSA: load %s' % self.train_list)
        self.__cur_id = 0  # for training
        self.id_list = []
        self.id_idx_list = []
        with open(self.train_list, 'r') as file:
            idx = 0
            for line in file:
                data = line.split(',')
                pano_id = (data[0].split('/')[-1]).split('.')[0]
                # satellite filename, streetview filename, pano_id
                self.id_list.append([data[0], data[1], pano_id])
                self.id_idx_list.append(idx)
                idx += 1
        self.data_size = len(self.id_list)
        if print_bool:
            print('CVUSA: load', self.train_list, ' data_size =', self.data_size)
            print('CVUSA: load %s' % self.test_list)
        self.__cur_test_id = 0  # for training
        self.id_test_list = []
        self.id_test_idx_list = []
        with open(self.test_list, 'r') as file:
            idx = 0
            for line in file:
                data = line.split(',')
                pano_id = (data[0].split('/')[-1]).split('.')[0]
                # satellite filename, streetview filename, pano_id
                self.id_test_list.append([data[0], data[1], pano_id])
                self.id_test_idx_list.append(idx)
                idx += 1
        self.test_data_size = len(self.id_test_list)
        if print_bool:
            print('CVUSA: load', self.test_list, ' data_size =', self.test_data_size)

        # if mode == 'train':
        #     self.neighborhood_id = gpsSample(location_list=root, id_list=self.id_list)

    def __getitem__(self, index, debug=False):
        if self.mode == 'train':
            idx = index % len(self.id_idx_list)
            img_query = Image.open(self.root + self.id_list[idx][1]).convert('RGB')
            img_reference = Image.open(self.root + self.id_list[idx][0]).convert('RGB')

            aug = random.choice([0, 1])
            if aug:
                # shift_angle = random.randint(1, 359)  # STTVCF
                shift_angle = random.choice([90, 180, 270])  # STTV
                img_query = self.transform_shift(size=self.grd_size, shift=shift_angle)(img_query)
                img_reference = self.transform_rotate(size=self.sat_size, angle=shift_angle)(img_reference)
            else:
                img_query = self.transform_query(img_query)
                img_reference = self.transform_reference(img_reference)

            return img_query, img_reference, torch.tensor(idx), torch.tensor(idx)

        elif 'test_reference' in self.mode:
            img_reference = Image.open(self.root + self.id_test_list[index][0]).convert('RGB')
            img_reference = self.transform_reference(img_reference)
            return img_reference, torch.tensor(index)

        elif 'test_query' in self.mode:
            img_query = Image.open(self.root + self.id_test_list[index][1]).convert('RGB')
            img_query = self.transform_query(img_query)
            return img_query, torch.tensor(index), torch.tensor(index)
        else:
            print('not implemented!!')
            raise Exception

    def __len__(self):
        if self.mode == 'train':
            return len(self.id_idx_list)
        elif 'test_reference' in self.mode:
            return len(self.id_test_list)
        elif 'test_query' in self.mode:
            return len(self.id_test_list)
        else:
            print('not implemented!')
            raise Exception
