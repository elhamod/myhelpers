# adopted from https://github.com/samaonline/Orthogonal-Convolutional-Neural-Networks/blob/master/imagenet/utils.py

import torch
import numpy as np

def deconv_orth_dist2(kernel1, kernel2, stride = 2, padding = 1):
    assert (kernel1.shape[0] == kernel2.shape[0]) or (kernel1.shape[1] == kernel2.shape[1]) , "Kernels should be of compatible sizes" + str(kernel1.shape) + ", " + str(kernel2.shape)
    if kernel1.shape[1] != kernel2.shape[1]:
        kernel1 = kernel1.permute(1, 0, 2, 3)
        kernel2 = kernel2.permute(1, 0, 2, 3)
    # [o_c, i_c, w, h] = kernel1.shape
    output = torch.conv2d(kernel1, kernel2, stride=stride, padding=padding)
    # target = torch.zeros((o_c, o_c, output.shape[-2], output.shape[-1])).cuda()
    # ct = int(np.floor(output.shape[-1]/2))
    # target[:,:,ct,ct] = torch.eye(o_c).cuda()
    return torch.norm(output)


def deconv_orth_dist(kernel, stride = 2, padding = 1):
    [o_c, i_c, w, h] = kernel.shape
    output = torch.conv2d(kernel, kernel, stride=stride, padding=padding)
    target = torch.zeros((o_c, o_c, output.shape[-2], output.shape[-1])).cuda()
    ct = int(np.floor(output.shape[-1]/2))
    target[:,:,ct,ct] = torch.eye(o_c).cuda()
    return torch.norm( output - target )
    
def orth_dist(mat, stride=None):
    mat = mat.reshape( (mat.shape[0], -1) )
    if mat.shape[0] < mat.shape[1]:
        mat = mat.permute(1,0)
    return torch.norm( torch.t(mat)@mat - torch.eye(mat.shape[1]).cuda())


# Example
# # compute output
# output = model(images)
# #####
# diff = utils.orth_dist(model.module.layer2[0].downsample[0].weight) + utils.orth_dist(model.module.layer3[0].downsample[0].weight) + utils.orth_dist(model.module.layer4[0].downsample[0].weight)
# diff += utils.deconv_orth_dist(model.module.layer1[0].conv1.weight, stride=1) + utils.deconv_orth_dist(model.module.layer1[1].conv1.weight, stride=1)
# diff += utils.deconv_orth_dist(model.module.layer2[0].conv1.weight, stride=2) + utils.deconv_orth_dist(model.module.layer2[1].conv1.weight, stride=1)
# diff += utils.deconv_orth_dist(model.module.layer3[0].conv1.weight, stride=2) + utils.deconv_orth_dist(model.module.layer3[1].conv1.weight, stride=1)
# diff += utils.deconv_orth_dist(model.module.layer4[0].conv1.weight, stride=2) + utils.deconv_orth_dist(model.module.layer4[1].conv1.weight, stride=1)
# #####
# loss = criterion(output, target) + args.r * diff