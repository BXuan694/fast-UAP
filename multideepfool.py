import numpy as np
from torch.autograd import Variable
import torch as torch
import copy
from torch.autograd.gradcheck import zero_gradients

def multideepfool(image, net, num_classes, overshoot, max_iter):
    """
       :param image: Image of size bx3xHxW
       :param net: network (input: images, output: values of activation **BEFORE** softmax).
       :param num_classes: num_classes (limits the number of classes to test against, by default = 10)
       :param overshoot: used as a termination criterion to prevent vanishing updates (default = 0.02).
       :param max_iter: maximum number of iterations for deepfool (default = 50)
       :return: minimal perturbation that fools the classifier, number of iterations that it required, new estimated_label and perturbed image
    """
    is_cuda = torch.cuda.is_available()
    if is_cuda:
        image = image.cuda()
        net = net.cuda()

    f_image = net.forward(Variable(image, requires_grad=True)).data.cpu().numpy()

    I = np.array([i.argsort()[::-1] for i in f_image])[:, 0:num_classes]
    label = I[:, 0]

    input_shape = image.cpu().numpy().shape
    pert_image = copy.deepcopy(image)
    w = np.zeros(input_shape, dtype=np.float32)
    r_tot = np.zeros(input_shape, dtype=np.float32)

    x = Variable(pert_image, requires_grad=True)
    fs = net.forward(x)

    k_i = label.copy()
    loop = np.zeros(f_image.shape[0], dtype=np.int32)
    for imgidx in range(f_image.shape[0]):
        while k_i[imgidx] == label[imgidx] and loop[imgidx] < max_iter:
            pert = np.inf
            fs[imgidx, I[imgidx, 0]].backward(retain_graph=True)
            grad_orig = x.grad.data.cpu().numpy()[imgidx].copy()

            for k in range(1, num_classes):
                zero_gradients(x)
                fs[imgidx, I[imgidx, k]].backward(retain_graph=True)

                cur_grad = x.grad.data.cpu().numpy()[imgidx].copy()
                w_k = cur_grad - grad_orig
                f_k = (fs[imgidx, I[imgidx, k]] - fs[imgidx, I[imgidx, 0]]).data.cpu().numpy()

                pert_k = abs(f_k)/np.linalg.norm(w_k.flatten())
                if pert_k < pert:
                    pert = pert_k
                    w = w_k

            r_i = (pert+1e-4) * w / np.linalg.norm(w)
            r_tot[imgidx] = np.float32(r_tot[imgidx] + r_i)

            if is_cuda:
                pert_image[imgidx] = image[imgidx] + (1+overshoot)*torch.from_numpy(r_tot[imgidx]).cuda()
            else:
                pert_image[imgidx] = image[imgidx] + (1+overshoot)*torch.from_numpy(r_tot[imgidx])

            x = Variable(pert_image, requires_grad=True)
            fs = net.forward(x)
            k_i[imgidx] = np.argmax(fs.data.cpu().numpy()[imgidx].flatten())

            loop[imgidx] += 1

    diff = label != k_i

    V = np.zeros([3, 224, 224], dtype=np.float32)
    for cond in range(diff.shape[0]):
        if diff[cond] == 1:
            V += r_tot[cond]
    return V*(1+overshoot), loop, label, k_i, pert_image

