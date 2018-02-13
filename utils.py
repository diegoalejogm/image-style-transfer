import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from torch.autograd import Variable
from torchvision import transforms
import torch
import os
import errno

IMAGES_PATH = './images'
imagenet_mean = [0.48501961, 0.45795686, 0.40760392] # IN RGB
# pr_imagenet_mean = [123.68000055, 116.7789993, 103.9389996] # IN RGB

def load_image(path, max_size=None, shape=None, display=True):
    '''
    Returns numpy array containing image data
    '''
    if max_size is not None and shape is not None:
        raise Exception('Either \'max_size\' or \'shape\' can be set.')

    img = Image.open('{}/{}'.format(IMAGES_PATH, path))

    # Resize image dimensions to same factor using max_size
    if max_size is not None:
        img.thumbnail(max_size, Image.ANTIALIAS)

    # Resize image dimensions to specified shape
    if shape is not None:
        img = img.resize(shape, Image.ANTIALIAS)

    # Transform image to tensor
    tensor = transforms.ToTensor()(img)

    # Display image
    if display:
        display_image(tensor)

    # Return tensor
    return tensor


def tensor_to_4d_var(tensor, requires_grad=False):
    '''
    Return image tensor with additional dimension N, as Variable
    '''
    tensor = tensor.unsqueeze(0)

    if torch.cuda.is_available():
        tensor = tensor.cuda()

    var = Variable(tensor, requires_grad=requires_grad)
    return var


def display_image(tensor, format='HWC'):
    '''
    Displays image from tensor using MatplotLib
    '''
    # Reset plot
    plt.close()
    plt.figure(dpi=100)

    if (len(tensor.size()) > 3):
        # Remove the batch dimension
        tensor = tensor.squeeze(0)
    # Load image
    img = transforms.ToPILImage()(tensor.clone().cpu())
    # Show image
    plt.imshow(img, aspect='auto')
    plt.show()


def gram_matrices(list):
    '''
    Computes the gram matrices of list of 4-D (NCWH) Convolved Tensors.
    '''
    new_list = []
    for tensor in list:
        new_list.append(_gram_matrix(tensor))
    return new_list


def _gram_matrix(tensor):
    '''
    Computes the gram matrix of a 4-D (NCWH) Convolved Tensor.
    '''
    # Batch size = 1 so it is skipped. N is num_feature_maps
    _, N, H, W = tensor.size()
    # Reshape tensor as feature map. New shape: N, M
    M = H * W
    tensor = tensor.view(N, M)
    # Calculate transposed tensor
    tensor_T = tensor.transpose(0, 1)
    # Calculate Gram matrix as tensor_T * tensor
    tensor = torch.mm(tensor, tensor_T)
    return tensor / (4. * (N*M)**2)


def content_loss(criterion, target_content, input_content):
    '''
    Calculates the content loss given a loss criterion
    '''
    
    loss = 1/2. * criterion(input_content[0], target_content[0])
    return loss


def preprocess(tensor):
    """
    Subtract ImageNet mean pixel-wise from a RGB image.
    Input Tensor must be in CWH format.
    """
    t = tensor.clone()
    t[0, :, :] -= imagenet_mean[0]
    t[1, :, :] -= imagenet_mean[1]
    t[2, :, :] -= imagenet_mean[2]
    return t


def postprocess(var):
    """
    Add ImageNet mean pixel-wise from a RGB image.
    Input Tensor must be in CWH format.
    """
    clip_preprocessed(var)
    var.data[:, 0, :, :] += imagenet_mean[0]
    var.data[:, 1, :, :] += imagenet_mean[1]
    var.data[:, 2, :, :] += imagenet_mean[2]


def clip_preprocessed(var):
    """
    Clips an image to the input values using the preprocessed mean.
    """
    min_val, max_val = 0., 1.
    var[:, 0, :, :].data.clamp_(min_val - imagenet_mean[0], max_val - imagenet_mean[0])
    var[:, 1, :, :].data.clamp_(min_val - imagenet_mean[1], max_val - imagenet_mean[1])
    var[:, 2, :, :].data.clamp_(min_val - imagenet_mean[2], max_val - imagenet_mean[2])


def style_loss(criterion, target_styles, input_styles):
    '''
    Calculates the style loss given a loss criterion
    '''
    loss = 0
    layer_weight = 1. / len(target_styles)
    for i, _ in enumerate(target_styles):
        input_i = input_styles[i]
        style_i = target_styles[i]
        loss_i = criterion(input_i, style_i) * layer_weight
        loss += loss_i
    return loss


def _save_image(tensor, step):
    out_dir = './data/images'
    _make_dir(out_dir)

    tensor = tensor.clone().cpu()[0]
    im = transforms.ToPILImage()(tensor)
    im.save('{}/step_{}.jpg'.format(out_dir, step))


def _make_dir(directory):
    try:
        os.makedirs(directory)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise
