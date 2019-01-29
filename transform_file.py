from torchvision import transforms
import numpy as np

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

cut = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    ])

convert = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std),
    ])

convert_pert = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0, 0, 0], std=std),
    ])

transform = transforms.Compose([
    cut.transforms[0],
    cut.transforms[1],
    convert.transforms[0],
    convert.transforms[1]
    ])
