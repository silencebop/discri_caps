import os
import sys
import random
from numpy import *
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import pandas as pd
import numpy as np

from PIL import Image
import cv2
import random


class Dataset(data.Dataset):

    def __init__(self, root, img_paths, img_labels, domain_labels, transform=None, get_aux=True, aux=None):
        """Load image paths and labels from gt_file"""
        self.root = root
        self.transform = transform
        self.get_aux = get_aux
        self.img_paths = img_paths
        self.img_labels = img_labels
        self.domain_labels = domain_labels
        self.aux = aux

    def __getitem__(self, idx):
        """Load image.

        Args:
            idx (int): image idx.
        Returns:
            img (tensor): image tensor

        """
        img_path1 = self.img_paths[idx][1]
        if idx < (len(self.img_paths) - 1):
            img_path2 = self.img_paths[idx + 1]
        img = Image.open(os.path.join(self.root, img_path1)).convert('RGB')
        label = self.img_labels[idx]
        domain_label = self.domain_labels[idx]

        if self.transform:
            img = self.transform(img)

        if self.get_aux:
            return img, label, domain_label

    # return img, label, self.aux[idx]

    def __len__(self):
        return len(self.img_paths)


class Dataset_flow(data.Dataset):

    def __init__(self, root, img_paths, img_labels, domain_labels, transform=None, get_aux=True, aux=None):
        """Load image paths and labels from gt_file"""
        self.root = root
        self.transform = transform
        self.get_aux = get_aux
        self.img_paths = img_paths
        self.img_labels = img_labels
        self.domain_labels = domain_labels
        self.aux = aux

    def __getitem__(self, idx):
        """Load image.

        Args:
            idx (int): image idx.

        Returns:
            img (tensor): image tensor

        """
        img_path1 = self.img_paths[idx][1]  # apex_frame
        img_path2 = self.img_paths[idx][0]  # onset_frame
        # print(img_path1)

        img1 = cv2.imread(os.path.join(self.root, img_path1), 0)  # 0: load the image as gray
        img2 = cv2.imread(os.path.join(self.root, img_path2), 0)  # 1: load the image as color
        img3 = cv2.imread(os.path.join(self.root, img_path2), 1)
        hsv = np.zeros_like(img3)
        hsv[..., 1] = 255
        # img1 = cv2.cvtColor(imgs1,cv2.COLOR_BGR2GRAY)
        # img2 = cv2.cvtColor(imgs2,cv2.COLOR_BGR2GRAY)

        # return a two-channels optical flow, actually it's displacement value of each point(wei yi)
        flow = cv2.calcOpticalFlowFarneback(img1, img2, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        # convert di cart coordinate to polar coordinate to get ji zhi, ji jiao
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        hsv[..., 0] = ang * 180 / np.pi / 2
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        PIL_image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)  # mode:hsv(array)-->RGB(array)

        img = Image.fromarray(PIL_image)  # ndarray-->Image
        label = self.img_labels[idx]
        domain_label = self.domain_labels[idx]

        if self.transform:
            img = self.transform(img)  # Image-->Tensor

        if self.get_aux:
            return img, label, domain_label

    # return img, label, self.aux[idx]

    def __len__(self):
        return len(self.img_paths)


def get_triple_meta_data(file_path):
    df = pd.read_csv(file_path)
    on_paths = list(df.on_frame_path)
    apex_paths = list(df.apex_frame_path)
    off_paths = list(df.off_frame_path)

    paths = [(on, apex, off) for (on, apex, off) in zip(on_paths, apex_paths, off_paths)]
    labels = list(df.label)
    return paths, labels


def get_triple_meta_data_f(df):  # san chong yuan data (a1,a2,a3)
    # df = pd.read_csv(file_path)
    # paths = list(df.apex_frame_path)
    on_paths = list(df.onset_frame_path)
    apex_paths = list(df.apex_frame_path)
    off_paths = list(df.offset_frame_path)

    paths = [(on, apex, off) for (on, apex, off) in zip(on_paths, apex_paths, off_paths)]
    class_labels = np.array(list(df.label))
    domain_labels = np.array(list(df.domain_label))
    return paths, class_labels, domain_labels


def get_meta_data(df):
    paths = list(df.apex_frame_path)
    labels = list(df.label)
    domain_labels = list(df.domain_label)
    return paths, labels, domain_labels


def data_split(file_path, subject_out_idx=0):
    """Split dataset into train set and validation set"""
    # data, subject, clipID, label, apex_frame, apex_frame_path
    data_sub_column = 'data_sub'

    df = pd.read_csv(file_path)  # read data_apex.csv and convert it to the type of DataFrame:
    # data (435, 11)  0~434
    df_val = df[df['domain_label'].values == 1]
    # print(type(df))
    subject_list = list(df_val[data_sub_column].unique())  # return all folder name with one time list of the column 'data
    # _sub' from data_apex.csv   eg: ['smic_20', 'smic_14', 'smic_18',....., 'samm_35', 'casme2_sub01']  list: 68
    # print(subject_list)
    # random.shuffle(subject_list)
    subject_out = subject_list[subject_out_idx]
    print('subject_out', subject_out)
    df_train = df[df[data_sub_column] != subject_out]  # de_train: the rest of df not include lines
    # corresponding to subject_out
    # df_val.sample(frac=0.3, replace=True, axis=0)
    df_val = df_val[df_val[data_sub_column] == subject_out]  # 留一法交叉验证
    # df_val = df_val[df_val['domain_label'].values != 0]
    return df_train, df_val


def upsample_subdata(df, df_four, number=4):
    result = df.copy()
    for i in range(df.shape[0]):
        quotient = number // 1  # shang (zhi qu zheng shu wei)
        remainder = number % 1  # yu shu
        remainder = 1 if np.random.rand() < remainder else 0
        value = quotient + remainder

        tmp = df_four[df_four['data_sub'] == df.iloc[i]['data_sub']]
        tmp = tmp[tmp['clip'] == df.iloc[i]['clip']]  # 取第i行的clip列

        value = min(value, tmp.shape[0])
        tmp = tmp.sample(int(value))
        result = pd.concat([result, tmp])
    return result


def sample_data(df, df_four):
    df_neg = df[df.label == 0]
    df_pos = df[df.label == 1]
    df_sur = df[df.label == 2]
    print('df_negr', df_neg.shape)
    print('df_posr', df_pos.shape)
    print('df_surr', df_sur.shape)

    num_sur = 4
    num_pos = 5 * df_sur.shape[0] / df_pos.shape[0] - 1
    if num_pos < 1:
        num_pos = 1

    # num_neg = 4
    num_neg = 5 * df_sur.shape[0] / df_neg.shape[0] - 1
    if num_neg < 1:
        num_neg = 0
    # print(num_neg)
    df_neg = upsample_subdata(df_neg, df_four, num_neg)
    df_pos = upsample_subdata(df_pos, df_four, num_pos)
    df_sur = upsample_subdata(df_sur, df_four, num_sur)
    print('df_neg', df_neg.shape)
    print('df_pos', df_pos.shape)
    print('df_sur', df_sur.shape)

    df = pd.concat([df_neg, df_pos, df_sur])  # 默认列数不变，行数发生变化（增加），类似于表的连接
    return df
