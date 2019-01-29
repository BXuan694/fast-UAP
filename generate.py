from multideepfool import multideepfool
import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from PIL import Image
import torch.backends.cudnn as cudnn
import sys
from transform_file import transform, cut, convert, std, convert_pert
from targetmodel import MyDataset
from torchvision import transforms

def project_lp(v, xi, p):
    if p == 2:
        l2 = np.linalg.norm(v)
        if l2 > xi:
            v = v/l2
    elif p == np.inf:
        v = np.sign(v)*np.minimum(abs(v), xi)
    else:
        raise ValueError("Projection function not found, please check.")
    return v

def generate(path, transet, testset, net, delta=0.2, max_iter_uni=np.inf, xi=0.2, p=np.inf, num_classes=10, overshoot=0.2, max_iter_df=50, batch_size=2):

    net.eval()
    if torch.cuda.is_available():
        device = 'cuda'
        net.cuda()
        cudnn.benchmark = True
    else:
        device = 'cpu'

    transet = os.path.join(path, transet)
    testset = os.path.join(path, testset)
    if not os.path.isfile(transet):
        print("Training data of UBP does not exist, please check!")
        sys.exit()
    if not os.path.isfile(testset):
        print("Testing data of UBP does not exist, please check!")
        sys.exit()

    v = np.zeros([224, 224, 3], dtype=np.float32)
    v_tensor = convert_pert(v)

    fooling_rate = 0.0
    iter = 0
    labels = open('./data/labels.txt', 'r').read().split('\n')

    # start an epoch
    while fooling_rate < 1-delta and iter < max_iter_uni:
        print("Starting pass number ", iter)
        train_data = MyDataset(txt=transet, transform=transform)
        train_dataloader = DataLoader(dataset=train_data, batch_size=batch_size, pin_memory=True, shuffle=True)

        for batch_idx, (inputs_orig, _) in enumerate(train_dataloader):
            inputs_pert = inputs_orig + v_tensor
            r1 = net(inputs_orig.to(device)).to('cpu').argmax(dim=1) # batch label
            r2 = net(inputs_pert.to(device)).to('cpu').argmax(dim=1)

            noWork_idx = (r1 == r2).nonzero()
            df_batchsize = noWork_idx.numel()

            print(">> batch =", batch_idx, " deepfool size:", df_batchsize, ', pass #', iter)
            if df_batchsize > 1:
                per_imgs = inputs_pert[noWork_idx].squeeze()
            elif df_batchsize == 1:
                per_imgs = inputs_pert[noWork_idx].squeeze()[None, :]

            if df_batchsize > 0:
                dr, _, _, _, _ = multideepfool(per_imgs, net, num_classes=num_classes, overshoot=overshoot, max_iter=max_iter_df)
                v_tensor += torch.Tensor(dr) 
                v_tensor = project_lp(v_tensor, xi, p)

        iter += 1
        v = v_tensor.numpy().transpose([2,1,0]) * std

        with torch.no_grad():
            # Compute fooling_rate
            est_labels_orig = torch.tensor(np.zeros(0, dtype=np.int64))
            est_labels_pert = torch.tensor(np.zeros(0, dtype=np.int64))

            batch = 32

            test_data_orig = MyDataset(txt=testset, transform=transform)
            test_data_pert = MyDataset(txt=testset, transform=transform, pert=v)
            test_loader_orig = DataLoader(dataset=test_data_orig, batch_size=batch, pin_memory=True)
            test_loader_pert = DataLoader(dataset=test_data_pert, batch_size=batch, pin_memory=True)

            for batch_idx, (inputs, _) in enumerate(test_loader_orig):
                inputs_orig = inputs.to(device)
                outputs = net(inputs)
                _, predicted = outputs.max(1)
                est_labels_orig = torch.cat((est_labels_orig, predicted.cpu()))
            torch.cuda.empty_cache()

            for batch_idx, (inputs, _) in enumerate(test_loader_pert):
                inputs_pert = inputs.to(device)
                outputs = net(inputs)
                _, predicted = outputs.max(1)
                est_labels_pert = torch.cat((est_labels_pert, predicted.cpu()))
            torch.cuda.empty_cache()

            fooling_rate = float(torch.sum(est_labels_orig != est_labels_pert))/len(est_labels_orig)
            print("FOOLING RATE: ", fooling_rate)
            np.save('v'+str(iter)+'_'+str(round(fooling_rate, 4)), v)

    return v
