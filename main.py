import matplotlib.pyplot as plt
import os
import numpy as np
import torch
from PIL import Image
import torch.backends.cudnn as cudnn
import sys
from transform_file import transform, cut
from targetmodel import ResNet_ft, VGG_ft, root
from generate import generate
import torchvision.models as models
import argparse

parser = argparse.ArgumentParser(description='fast universal adversarial perturbation')
parser.add_argument('-p', action="store", dest="PATH", default=root, help='Path of data set')
parser.add_argument('-n', action="store", dest="n", default='r', help='Choose a network, "r" for res50 and "v" for vgg16')
parser.add_argument('-b', action="store", dest="batch_size", default=2, help="Choose batch size. 2, 4 and 8 recommended", type=int)
parser.add_argument('-m', action="store", dest="xi", default=0.16, help="Magnitude of adversarial perturbation, 0.15 recommended", type=float)
args = parser.parse_args()

if args.batch_size < 1:
    print("batch size err, please check.")
    sys.exit()
if args.xi < 0:
    print("magnitude err, please check.")
    sys.exit()

print('>> Loading network...')
if args.n == 'r':
    resnet50 = models.resnet50(pretrained=False)
    net = ResNet_ft(resnet50)
    loadfile = torch.load('./checkpoint/resnet50-0.864591.t7', map_location=lambda storage, loc: storage)
    name = 'ResNet50'
elif args.n == 'v':
    vgg = models.vgg16(pretrained=False)
    net = VGG_ft(vgg)
    loadfile = torch.load('./checkpoint/vgg16-ckpt0.742412.t7', map_location=lambda storage, loc: storage)
    name = 'VGG16'
else:
    print("Network not found, please check.")
    sys.exit()
net.load_state_dict(loadfile['net'])
acc = loadfile['acc']
net.eval()
print('   Finished loading ' + name + ' of accuracy ' + str(acc))

print('>> Checking devices...')
if torch.cuda.is_available():
    device = 'cuda'
    net.cuda()
    cudnn.benchmark = True
else:
    device = 'cpu'
print('   Found ' + device + '.')

print('>> Loading perturbation...')
# generate perturbation v of 224*224*3 of [-10,10] directly on original image.
file_perturbation = 'data/universal.npy'
if os.path.isfile(file_perturbation) == 0:
    print('   No perturbation found, computing...')

    print('>> Checking dataset...')
    if not os.path.exists(args.PATH):
        print("Data set not found. please check!")
        sys.exit()
    print('   Done.')
    v = generate(args.PATH, 'dataset4u-trn.txt', 'dataset4u-val.txt', net, max_iter_uni=10, delta=0.1, p=np.inf, num_classes=25, overshoot=0.1, max_iter_df=500, xi=args.xi, batch_size=args.batch_size)
    # Saving the universal perturbation
    np.save('./data/universal.npy', v)
else:
    print('   Found a pre-computed universal perturbation at', file_perturbation)
    v = np.load(file_perturbation)


testimg = "./data/test_im4.jpg"
print('>> Testing the universal perturbation on', testimg)
labels = open('./data/labels.txt', 'r').read().split('\n')
testimgToInput = Image.open(testimg).convert('RGB')
pertimgToInput = np.clip(cut(testimgToInput)+v, 0, 255)
pertimg = Image.fromarray(pertimgToInput.astype(np.uint8))

img_orig = transform(testimgToInput)
inputs_orig = img_orig[np.newaxis, :].to(device)
outputs_orig = net(inputs_orig)
_, predicted_orig = outputs_orig.max(1)
label_orig = labels[predicted_orig[0]]

img_pert = transform(pertimg)
inputs_pert = img_pert[np.newaxis, :].to(device)
outputs_pert = net(inputs_pert)
_, predicted_pert = outputs_pert.max(1)
label_pert = labels[predicted_pert[0]]

# Show original and perturbed image
plt.figure()
plt.subplot(1, 2, 1)
plt.imshow(cut(testimgToInput), interpolation=None)
plt.title(label_orig)

plt.subplot(1, 2, 2)
plt.imshow(pertimg, interpolation=None)
plt.title(label_pert)

plt.savefig("./data/result.png")
plt.show()
