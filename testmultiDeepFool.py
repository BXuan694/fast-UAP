import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision.models as models
from PIL import Image
from multideepfool import multideepfool
import os
from transform_file import transform, mean, std
import torch.backends.cudnn as cudnn

net = models.resnet34(pretrained=True)
if torch.cuda.is_available():
    device = 'cuda'
    net.cuda()
    cudnn.benchmark = True
else:
    device = 'cpu'

net.eval()
labels = open(os.path.join('./data/synset_words.txt'), 'r').read().split('\n')

def clip_tensor(A, minv, maxv):
    A = torch.max(A, minv * torch.ones(A.shape))
    A = torch.min(A, maxv * torch.ones(A.shape))
    return A

clip = lambda x: clip_tensor(x, 0, 255)

im1 = transform(Image.open('./data/test_im1.jpeg').convert('RGB'))
im2 = transform(Image.open('./data/test_im2.jpg').convert('RGB'))
im3 = transform(Image.open('./data/test_im3.jpg').convert('RGB'))
im4 = transform(Image.open('./data/test_im4.jpg').convert('RGB'))

imgs = torch.Tensor(4, 3, 224, 224)
imgs[0] = im1
imgs[1] = im2
imgs[2] = im3
imgs[3] = im4

tf = transforms.Compose([transforms.Normalize(mean=[0, 0, 0], std=map(lambda x: 1 / x, mean)),
                        transforms.Normalize(mean=map(lambda x: -x, std), std=[1, 1, 1]),
                        transforms.Lambda(clip),
                        transforms.ToPILImage(),
                        transforms.CenterCrop(224)])

if not os.path.isfile("./data/demo_multiDeepFool.npy"):
    mdf, loop, label_orig, label_pert, pert_image = multideepfool(imgs, net, num_classes=20, overshoot=0.25, max_iter=100)
    np.save("./data/demo_multiDeepFool.npy", mdf)
    """
    mdf: 3*224*224
    LOOP: 4
    label_orig: 4
    label_pert: 4
    pert_image: 4*3*224*224
    """
    for i in range(label_orig.shape[0]):
        str_label_orig = labels[np.int(label_orig[i])].split(',')[0]
        str_label_pert = labels[np.int(label_pert[i])].split(',')[0]
        print("Original label["+str(i)+"] = ", str_label_orig)
        print("Perturbed label["+str(i)+"] = ", str_label_pert)