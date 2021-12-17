import os
import sys
import torch
import pickle

'''method improved'''
from torch.optim import Adam, lr_scheduler

from capsule.modules import MECapsuleNet
from capsule.loss import me_loss
from capsule.evaluations import Meter
from capsule.data import data_split, sample_data, get_meta_data, Dataset, get_triple_meta_data_f, Dataset_flow

from torchvision import transforms

from tqdm import tqdm
import pandas as pd
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix
from visdom import Visdom

'''==========================111111    start====================='''
sys.path.append(os.getcwd())
# data_apex_frame_path = 'datasets/data_apex.csv'
# data_four_frames_path = 'datasets/data_four_frames.csv'
# data_root = '/cis/staff/xiezhihua/anaconda3/envs/py37_pytorch/me_recognition/'
# data_root = '/home/bop/PycharmProjects/coding_capsule/'
data_root = os.getcwd()
data_apex_frame_path_train = 'datasets/data_all.csv'
# data_apex_frame_path_val = 'datasets/data_apex.csv'
# data_apex_frame_path = 'datasets/data1_apex.csv'
# data_apex_frame_path_train = 'datasets/data_apex.csv'

batch_size = 32
lr = 0.0001  # origin: 0.0001+6
# discri_lr = 0.01
lr_decay_value = 0.9
num_classes = 3
epochs = 30  # origin: 30
xishu = 0.1
NUM = 68

x_meter = Meter()  # to record some statistics
batches_scores = []


def load_me_data(data_root, file_path, subject_out_idx, batch_size=batch_size, num_workers=4):
    # 留一法交叉验证
    df_train, df_val = data_split(file_path, subject_out_idx)  # df_train: (413,11) df_val: (22, 11)
    # df_four = pd.read_csv(data_four_frames_path)
    # df_train_sampled = sample_data(df_train, df_four)  # 重采样技术  imbalanced class
    # df_train_sampled = shuffle(df_train_sampled)
    df_train_sampled = shuffle(df_train)

    # train_paths, train_labels = get_meta_data(df_train_sampled)
    train_paths, train_labels, train_domain_labels = get_triple_meta_data_f(df_train_sampled)

    train_transforms = transforms.Compose([transforms.Resize((234, 240)),
                                           transforms.RandomRotation(degrees=(-8, 8)),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ColorJitter(brightness=0.2, contrast=0.2,
                                                                  saturation=0.2, hue=0.2),
                                           transforms.RandomCrop((224, 224)),
                                           transforms.ToTensor()])

    train_dataset = Dataset_flow(root=data_root,
                                 img_paths=train_paths,
                                 img_labels=train_labels,
                                 domain_labels=train_domain_labels,
                                 transform=train_transforms)
    # train_dataset = Dataset(root=data_root,
    #                         img_paths=train_paths,
    #                         img_labels=train_labels,
    #                         domain_labels=train_domain_labels,
    #                         transform=train_transforms)

    val_transforms = transforms.Compose([transforms.Resize((234, 240)),
                                         transforms.RandomRotation(degrees=(-8, 8)),
                                         transforms.CenterCrop((224, 224)),
                                         transforms.ToTensor()])

    # val_paths, val_labels = get_meta_data(df_val)
    val_paths, val_labels, val_domain_labels = get_triple_meta_data_f(df_val)
    val_dataset = Dataset_flow(root=data_root,
                               img_paths=val_paths,
                               img_labels=val_labels,
                               domain_labels=val_domain_labels,
                               transform=val_transforms)
    # val_dataset = Dataset(root=data_root,
    #                       img_paths=val_paths,
    #                       img_labels=val_labels,
    #                       domain_labels=val_domain_labels,
    #                       transform=val_transforms)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               num_workers=num_workers,
                                               shuffle=True)

    val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                             batch_size=batch_size,
                                             num_workers=num_workers,
                                             shuffle=False)
    return train_loader, val_loader


def weight_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        torch.nn.init.xavier_normal_(m.weight.data)
        # torch.nn.init.kaiming_normal_(m.weight.data)
        torch.nn.init.constant_(m.bias.data, 0.0)


def on_epoch(model, optimizer, lr_decay, train_loader, test_loader, epoch):
    #  =========== 训练 ============
    model.train()
    # lr_decay.step()  # decrease the learning rate by multiplying a factor `gamma`
    train_loss = 0.0
    correct = 0.
    meter = Meter()

    # viz.line([0.], [0.], win='Train_loss', opts=dict(title='train_loss'))
    # viz.line([0.], [0.], win='Train_accuracy', opts=dict(title='train_acc'))
    # viz.line([0.], [0.], win='Test_loss', opts=dict(title='test_loss'))
    # viz.line([0.], [0.], win='Test_accuracy', opts=dict(title='test_acc'))

    steps = len(train_loader.dataset) // batch_size + 1

    with tqdm(total=steps) as progress_bar:  # jin du tiao
        # print('hello')
        for i, (x, y, y_domain) in enumerate(train_loader):  # batch training
            # print('changdu：', len(train_loader))

            y = torch.zeros(y.size(0), num_classes).scatter_(1, y.view(-1, 1),
                                                             1.)  # change to one-hot coding   [32,3]
            y_domain = torch.zeros(y_domain.size(0), 2).scatter_(1, y_domain.view(-1, 1),
                                                                 1.)  # change to one-hot coding   32 -> [32,2]
            x, y, y_domain = x.cuda(), y.cuda(), y_domain.cuda()  # convert input data to GPU Variable

            optimizer.zero_grad()  # set gradients of optimizer to zero
            y_pred, y_domain_pred = model(x, y, y_domain)  # forward   [ 32, 3]   [32,2]
            loss = me_loss(y, y_pred)  # compute loss
            domain_loss = 0
            domain_loss = torch.nn.BCEWithLogitsLoss()(y_domain_pred.squeeze(), y_domain.squeeze().float())
            # domain_loss/=batch_size
            print('train_domain_loss:', domain_loss.item(), 'train_loss:', loss.item())
            # xishu = xishu_set(loss, domain_loss)
            loss = loss - xishu * domain_loss
            # print('系数后loss:', loss, 'xishu:', xishu)
            loss.backward()  # backward, compute all gradients of loss w.r.t all Variables
            train_loss += loss.item() * x.size(0)  # record the batch loss
            optimizer.step()  # update the trainable parameters with computed gradients

            # y_pred = y_pred.data.max(1)[1]
            y_pred = y_pred.data.argmax(dim=1)  # return the order of a specific dim where its value is maximum
            y_true = y.data.max(1)[1]  # tensor([2, 1, 2, 0, 1, 1, 2, 2, 2, 0, 1, 0, 1, 1, 1, 1, 2, 1, 2, 2, 1, 2, 1,
            # 2,1, 1, 1, 2, 2, 0, 2, 2], device='cuda:0')

            meter.add(y_true.cpu().numpy(), y_pred.cpu().numpy())
            correct += y_pred.eq(y_true).cpu().sum()

            progress_bar.set_postfix(loss=loss.item(), correct=correct)
            progress_bar.update(1)

        train_loss /= float(len(train_loader.dataset))
        train_acc = float(correct.item()) / float(len(train_loader.dataset))
        scores = meter.value()
        meter.reset()
        print('Training UAR: %.4f' % (scores[0].mean()), scores[0])
        print('Training UF1: %.4f' % (scores[1].mean()), scores[1])
        # viz.line([train_loss], [steps], win='Train_loss', update='append')
        # viz.line([train_acc], [steps], win='Train_accuracy', update='append')

    lr_decay.step()
    #  =========== 验证 ============

    correct = 0.
    test_loss = 0.
    model.eval()
    for i, (x, y, y_domain) in enumerate(test_loader):  # batch training
        y = torch.zeros(y.size(0), num_classes).scatter_(1, y.view(-1, 1),
                                                         1.)  # change to one-hot coding
        y_domain = torch.zeros(y_domain.size(0), 2).scatter_(1, y_domain.view(-1, 1),
                                                             1.)
        x, y, y_domain = x.cuda(), y.cuda(), y_domain.cuda()  # convert input data to GPU Variable

        y_pred, y_domain_pred = model(x, y, y_domain)  # forward
        loss = me_loss(y, y_pred)  # compute loss
        domain_loss = torch.nn.BCEWithLogitsLoss()(y_domain_pred.squeeze(), y_domain.squeeze().float())
        domain_loss/=batch_size
        print('val_domain_loss:', domain_loss.item(), 'val_loss:', loss.item())
        # xishu = xishu_set(loss, domain_loss)
        loss = loss - xishu * domain_loss
        test_loss += loss.item() * x.size(0)  # record the batch loss

        y_pred = y_pred.data.max(1)[1]
        y_true = y.data.max(1)[1]

        meter.add(y_true.cpu().numpy(), y_pred.cpu().numpy())
        correct += y_pred.eq(y_true).cpu().sum()

        if (epoch + 1) % 10 == 0 and i % steps == 0:
            print('y_true\n', y_true[:30])
            print('y_pred\n', y_pred[:30])

    print('y_true', y.sum(dim=0))
    scores = meter.value()
    print('Testing UAR: %.4f' % (scores[0].mean()), scores[0])
    print('Testing UF1: %.4f' % (scores[1].mean()), scores[1])

    test_loss /= float(len(test_loader.dataset))
    test_acc = float(correct.item()) / float(len(test_loader.dataset))
    # viz.line([test_loss], [steps], win='Test_loss', update='append')
    # viz.line([test_acc], [steps], win='Test_accuracy', update='append')

    # last_test_acc = test_acc
    # last_test_loss = test_loss
    return train_loss, train_acc, test_loss, test_acc, meter


def train_eval(subject_out_idx):
    best_val_uf1 = 0.0
    best_val_uar = 0.0
    viz.line([0.], [0.], win='Train_loss1', opts=dict(title='train_loss1'))
    viz.line([0.], [0.], win='Train_accuracy1', opts=dict(title='train_acc1'))
    viz.line([0.], [0.], win='Test_loss1', opts=dict(title='test_loss1'))
    viz.line([0.], [0.], win='Test_accuracy1', opts=dict(title='test_acc1'))

    # Model & others
    model = MECapsuleNet(input_size=(3, 224, 224), classes=num_classes, routings=3, is_freeze=False)
    model.cuda()
    model.discriminator.apply(weight_init)
    # model.load_state_dict(torch.load("./scores_capsule_resnet_sampled_fer_freeze.pkl"))
    # model.load_state_dict(torch.load("/cis/staff2/xiezhihua/anaconda3/envs/py37_pytorch/me_recognition/scores_capsule_resnet_sampled_fer_freeze.pkl")['model'])
    optimizer = Adam(model.parameters(), lr=lr)  # origin:Adam
    # params = optimizer_params_set(model, lr, discri_lr)
    # optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    lr_decay = lr_scheduler.ExponentialLR(optimizer, gamma=lr_decay_value)  # zhishu shuai jian lr

    for epoch in range(epochs):
        train_loader, test_loader = load_me_data(data_root, data_apex_frame_path_train,
                                                 subject_out_idx=subject_out_idx,
                                                 batch_size=batch_size)
        train_loss, train_acc, test_loss, test_acc, meter = on_epoch(model, optimizer, lr_decay,
                                                                     train_loader, test_loader,
                                                                     epoch)

        print("==> Subject out: %02d - Epoch %02d: loss=%.5f, train_acc=%.5f, val_loss=%.5f, "
              "val_acc=%.4f"
              % (subject_out_idx, epoch, train_loss, train_acc,
                 test_loss, test_acc))
        viz.line([train_loss], [epoch], win='Train_loss1', update='append')
        viz.line([train_acc], [epoch], win='Train_accuracy1', update='append')
        viz.line([test_loss], [epoch], win='Test_loss1', update='append')
        viz.line([test_acc], [epoch], win='Test_accuracy1', update='append')

        scores = meter.value()
        if scores[1].mean() >= best_val_uf1:
            best_val_uar = scores[0].mean()
            best_val_uf1 = scores[1].mean()
            Y_true = meter.Y_true.copy()
            Y_pred = meter.Y_pred.copy()

    x_meter.add(Y_true, Y_pred, verbose=True)

    return best_val_uar, best_val_uf1


if __name__ == '__main__':
    viz = Visdom(env='my_dis')
    for i in range(1):  # 68 data_subs in total from 3 datasets + 98 data_subs from CK+
        scores = train_eval(subject_out_idx=i)
        batches_scores.append(scores)
        x_scores = x_meter.value()
        print('final uar', x_scores[0], x_scores[0].mean())
        print('final uf1', x_scores[1], x_scores[1].mean())
        print('---- NEXT ---- \n\n')

        with open('scores_capsule_resnet_sampled_fer_freeze.pkl', 'wb') as file:
            data = dict(meter=x_meter, batches_scores=batches_scores)
            pickle.dump(data, file)  # 序列化对象，将对象data保存到file中去
'''==========================111111    end====================='''
