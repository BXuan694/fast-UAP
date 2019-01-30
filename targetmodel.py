import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
from transform_file import cut

root = '/home/wang/Dataset/Caltech256/'

def default_loader(path):
    return Image.open(path).convert('RGB')

class MyDataset(Dataset):
    def __init__(self, txt, transform=None, pert=np.zeros(1), loader=default_loader):
        fh = open(txt, 'r')
        imgs = []
        for line in fh:
            line = line.strip()
            words = line.split()
            imgs.append((words[0], int(words[1])))
        self.imgs = imgs
        self.transform = transform
        self.loader = loader
        self.pert = pert

    def __getitem__(self, index):
        '''
        return Tensor with v
        '''
        fn, label = self.imgs[index]
        img = Image.fromarray(np.clip(cut(self.loader(fn))+self.pert, 0, 255).astype(np.uint8))
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.imgs)


class ResNet_ft(nn.Module):
    def __init__(self, model):
        super(ResNet_ft, self).__init__()

        self.resnet_layer = nn.Sequential(*list(model.children())[:-1])
        self.Linear_layer = nn.Linear(2048, 257)

    def forward(self, x):
        x = self.resnet_layer(x)
        x = x.view(x.size(0), -1)
        x = self.Linear_layer(x)
        return x


class VGG_ft(nn.Module):
    def __init__(self, model):
        super(VGG_ft, self).__init__()

        self.feature_layer = nn.Sequential(*list(model.children())[:-1])
        self.classifier_layer = nn.Sequential(
            nn.Linear(25088, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 257),
        )

    def forward(self, x):
        x = self.feature_layer(x)
        x = x.view(x.size(0), -1)
        x = self.classifier_layer(x)
        return x
