from torchvision import transforms
import numpy as np

def MakeSquared(img, imageDimension=256):

    img_H = img.size[0]
    img_W = img.size[1]

    # Resize and pad
    smaller_dimension = 0 if img_H < img_W else 1
    larger_dimension = 1 if img_H < img_W else 0
    if (imageDimension != img_H or imageDimension != img_W):
        new_smaller_dimension = int(imageDimension * img.size[smaller_dimension] / img.size[larger_dimension])
        if smaller_dimension == 1:
            img = transforms.functional.resize(img, (new_smaller_dimension, imageDimension))
        else:
            img = transforms.functional.resize(img, (imageDimension, new_smaller_dimension))

        diff = imageDimension - new_smaller_dimension
        pad_1 = int(diff/2)
        pad_2 = diff - pad_1
        mean = np.asarray([ 0.485, 0.456, 0.406 ])
        fill = tuple([int(round(mean[0]*255)), int(round(mean[1]*255)), int(round(mean[2]*255))])

#         print(img, pad_1, pad_2, fill)
        if smaller_dimension == 0:
            img = transforms.functional.pad(img, (pad_1, 0, pad_2, 0), padding_mode='constant', fill = fill)
        else:
            img = transforms.functional.pad(img, (0, pad_1, 0, pad_2), padding_mode='constant', fill = fill)

    return img
