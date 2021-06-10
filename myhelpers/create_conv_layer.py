import math
import torch
import collections

def get_output_res(input_res, kernel_size, stride, padding):
    return math.floor((input_res + 2*padding - kernel_size)/stride +1)

# Create an Conv layer with RELU and BatchNormalization
def get_conv(input_res, output_res, input_num_of_channels, intermediate_num_of_channels, output_num_of_channels, num_of_layers = 1, kernel_size=3, stride=1, padding=1):
    #  Sanity checks 
    assert(input_res >= output_res)
    needed_downsampling_layers=0
    res = input_res
    for i in range(num_of_layers):
        intermediate_output_res = get_output_res(res, kernel_size, stride, padding)
        assert(intermediate_output_res <= res)
        needed_downsampling_layers = needed_downsampling_layers + 1
        res = intermediate_output_res
        if intermediate_output_res == output_res:
            break

    l = [] 
    
    # First k layers no downsampling
    in_ = input_num_of_channels
    for i in range(num_of_layers - needed_downsampling_layers):
        out_ = intermediate_num_of_channels if i<num_of_layers-1 else output_num_of_channels
        l.append(('conv'+str(i), torch.nn.Conv2d(in_, out_, kernel_size=1, stride=1, padding=0, bias=False)))
        l.append(('bnorm'+str(i), torch.nn.BatchNorm2d(out_)))
        l.append(('relu'+str(i), torch.nn.ReLU()))
        in_ = out_

    # Then downsample each remaining layer till we get to the desired output resolution 
    for i in range(needed_downsampling_layers):
        out_ = output_num_of_channels if i + num_of_layers - needed_downsampling_layers == num_of_layers-1 else intermediate_num_of_channels
        l.append(('conv'+str(i+needed_downsampling_layers), torch.nn.Conv2d(in_, out_, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)))
        l.append(('bnorm'+str(i+needed_downsampling_layers), torch.nn.BatchNorm2d(out_)))
        l.append(('relu'+str(i+needed_downsampling_layers), torch.nn.ReLU()))
        in_ = out_

    d = collections.OrderedDict(l)
    seq = torch.nn.Sequential(d)
    
    return seq