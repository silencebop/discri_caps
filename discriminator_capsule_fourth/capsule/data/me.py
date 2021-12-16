import torch
import pandas as pd
from sklearn.utils import shuffle
from torchvision import transforms
from .dataset import data_split, sample_data, get_meta_data, Dataset

data_apex_frame_path = 'datasets/data_apex.csv'
data_four_frames_path = 'datasets/data_four_frames.csv'
data_root = '/cis/staff/xiezhihua/anaconda3/envs/py37_pytorch/me_recognition/'


def load_me_data(data_root, file_path, subject_out_idx, batch_size=32, num_workers=4):
    df_train, df_val = data_split(file_path, subject_out_idx)
    df_four = pd.read_csv(data_four_frames_path)
    df_train_sampled = sample_data(df_train, df_four)
    df_train_sampled = shuffle(df_train_sampled)

    train_paths, train_labels, domain_labels = get_meta_data(df_train_sampled)

    train_transforms = transforms.Compose([transforms.Resize((234, 240)),
                                           transforms.RandomRotation(degrees=(-8, 8)),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ColorJitter(brightness=0.2, contrast=0.2,
                                                                  saturation=0.2, hue=0.2),
                                           transforms.RandomCrop((224, 224)),
                                           transforms.ToTensor()])

    train_dataset = Dataset(root=data_root,
                            img_paths=train_paths,
                            img_labels=train_labels,
                            transform=train_transforms)

    val_transforms = transforms.Compose([transforms.Resize((234, 240)),
                                         transforms.RandomRotation(degrees=(-8, 8)),
                                         transforms.CenterCrop((224, 224)),
                                         transforms.ToTensor()])

    val_paths, val_labels = get_meta_data(df_val)

    val_dataset = Dataset(root=data_root,
                          img_paths=val_paths,
                          img_labels=val_labels,
                          transform=val_transforms)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               num_workers=num_workers,
                                               shuffle=True)

    val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                             batch_size=batch_size,
                                             num_workers=num_workers,
                                             shuffle=False)
    return train_loader, val_loader
