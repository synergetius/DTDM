import torch
import torch.nn.functional as F
from torch import nn
from torchvision.transforms import Compose, Resize, RandomCrop, CenterCrop, ToTensor, RandomHorizontalFlip
from torch.utils.data import Dataset, DataLoader
from PIL import Image 
import os
import models

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib
import numpy as np
import pandas as pd
import cv2
device = torch.device("cuda")
class TinyImageNet(Dataset):
    N_CLASS = 20 
    N_TRAIN = 500 
    N_VAL = 50 
    def __init__(self, root, train=True, transform=None, debug=False):
        super().__init__()
        self.root = root
        self.train = train
        self.transform = transform
        self.debug = debug

        if self.train:
            self.data_dir = os.path.join(self.root, 'train') if 'train' not in self.root else self.root
        else:
            self.data_dir = os.path.join(self.root, 'val', 'images')
            self.annotation_file = os.path.join(self.root, 'val', 'val_annotations.txt')

        self.samples = []
        if self.train:
            print(f"[DEBUG] 开始扫描训练集类别...")  
            self.classes = sorted(os.listdir(self.data_dir))[:self.N_CLASS]
            self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
            print(f"[DEBUG] 找到 {len(self.classes)} 个类别")  
            sample_ind = os.path.join(self.root, 'train_samples.csv')

            print('classes number:', len(self.classes))
            
            sample_count = np.zeros(len(self.classes))

            for i, cls in enumerate(self.classes):
                cls_dir = os.path.join(self.data_dir, cls, 'images')
                cls_idx = self.class_to_idx[cls]
                ls = sorted(os.listdir(cls_dir)) 
                ls = ls[:self.N_TRAIN]
                sample_count[i] += len(ls)
                for img_name in ls: 
                    self.samples.append((os.path.join(cls_dir, img_name), cls_idx))
            plot = False
            if plot:
                plt.bar(self.classes, sample_count)
                plt.savefig('train_samples.png', dpi=300, bbox_inches='tight')
        else:
            print(f"[DEBUG] 开始读取验证集注释文件: {self.annotation_file}")  
            with open(self.annotation_file, 'r') as f:
                lines = f.readlines()

            self.class_to_img = {}
            for line in lines:
                img, cls = line.split('\t')[:2]
                self.class_to_img.setdefault(cls, []).append(img)
            
            self.classes = sorted(list(self.class_to_img.keys()))[:self.N_CLASS] ##########
            self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
            print('validation classes number:', len(self.classes))
            sample_count = np.zeros(len(self.classes))
            
            
            self.samples = []
            for cls in self.classes:
                imgs = self.class_to_img[cls]
                for img in imgs[:self.N_VAL]:
                    sample_count[self.class_to_idx[cls]] += 1
                    self.samples.append((os.path.join(self.data_dir, img), self.class_to_idx[cls]))
            plot = False
            if plot:
                plt.bar(self.classes, sample_count)
                plt.savefig('val_samples.png', dpi=300, bbox_inches='tight')

    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img_ = self.transform(img)
        else:
            img_ = img
        return img_.to(device), torch.tensor(label, dtype = torch.int64).to(device), img

MODEL_PATH = "log/" + "tpm_20260112194027.pt"
model = models.tpm_xxt(num_classes = TinyImageNet.N_CLASS)
model.load_state_dict(torch.load(MODEL_PATH))
model = model.to(device)

def custom_normalize(tensor):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1) 
    return (tensor - mean) / std
val_dataset = TinyImageNet(
    root = '/mnt/d/data/tiny-imagenet-200',
    train = False,
    transform = Compose([
        Resize(64),
        ToTensor(),
        custom_normalize
    ]) 
)
masks = []
rects = []
def save_mask(module, input, output):
    masks.append(output)
def save_rects(module, input, output):
    rects.append(output[1][-1])
for i in range(model.num_unfold_layers):
    model.admit_layers[i].register_forward_hook(save_rects)
    model.admit_layers[i].project_mask.register_forward_hook(save_mask)
print(model.query_rect) ################### 训练时被改变了，向中心收缩，可能不利于下游任务的泛化，所以要考虑固定一下
# /mnt/d/programming/DTDM
# conda activate dtdm-env
# python visualize_tpm.py
for ind in range(10, 20):#range(200):
    ind = ind * 5
    print(ind)
    input, target, img = val_dataset[ind]
    input = input.unsqueeze(0)
    target = target.unsqueeze(0)
    masks.clear()
    rects.clear()
    output = model(input)
    # plt.figure(figsize = (50, 15))
    
    n = len(masks)
    fig, axs = plt.subplots(1, n, figsize=(200, 50))
    # plt.subplot(1, n, 1)
    axs[0].imshow(img)
    axs[0].axis('off')
    for i, (m, r) in enumerate(zip(masks, rects)):
        # plt.subplot(1, n, i + 2)
        _, __, H, W = input.shape
        axs[i].imshow(img, extent=[0 - 0.5, H - 0.5, 0 - 0.5, W - 0.5])
        axs[i].axis('off')
        m = m.view(m.shape[0], m.shape[1] * model.admit_layers[i].num_rects, model.admit_layers[i].groups)
        r = r.view(r.shape[0], r.shape[1] * r.shape[2], r.shape[3])
        
        r = r.squeeze()
        
        fac = torch.tensor([H - 1, W - 1, H - 1, W - 1]).view(1, 4).to(r.device)
        # print(r.shape, fac.shape)
        # print(r) ############ 有可能query_rect被训练改变了？？
        r = r * fac
        m = torch.sum(m, dim = 2).squeeze() # 暂时用直接相加的方法
        m = m - torch.min(m)
        m = m / torch.max(m)
        colors = cv2.applyColorMap(np.uint8(m.cpu().detach().numpy()*255), cv2.COLORMAP_JET)
        colors = cv2.cvtColor(colors, cv2.COLOR_BGR2RGB) / 255
        for cj, rj in zip(colors, r):
            # print(cj)
            rect_patch = patches.Rectangle(
                (rj[0].item(), rj[1].item()), rj[2].item() - rj[0].item(), rj[3].item() - rj[1].item(),
                linewidth=2,
                edgecolor=cj,
                facecolor=cj,
                alpha=0.5  # 设置透明度
            )
            
            # 将矩形添加到坐标轴
            axs[i].add_patch(rect_patch)
    plt.savefig(f'visualization/tpm_{ind}.png', bbox_inches = 'tight')
    