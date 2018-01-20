import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from torch.autograd import Variable
from torchvision import transforms
import torch

IMAGES_PATH = './images'


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


def tensor_to_4d_var(tensor, requires_grad=False, cuda=False):
    '''
    Return image tensor with additional dimension N, as Variable
    '''
    var = Variable(tensor.unsqueeze(0), requires_grad=requires_grad)

    if cuda:
        var = var.cuda()
    return var


def display_image(tensor, format='HWC'):
    '''
    Displays image from tensor using MatplotLib
    '''
    # Reset plot
    plt.close()
    plt.figure()

    if (len(tensor.size()) > 3):
        # Remove the batch dimension
        tensor = tensor.squeeze(0)
    # Load image
    img = transforms.ToPILImage()(tensor.clone().cpu())
    # Show image
    plt.imshow(img)
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
    # Here M is minibatch size
    # Here N is num_feature_maps
    M, N, W, H = tensor.size()
    # Merge height and width dimensions into M: new shape: NCM
    tensor = tensor.view(M * N, W * H)
    # Calculate transposed tensor
    tensor_T = tensor.transpose(0, 1)
    # Calculate Gram matrix as tensor_T * tensor
    tensor = torch.mm(tensor, tensor_T)
    # print(C * W * H)
    return tensor / (4 * N**2 * (W * H)**2)


def style_loss(criterion, target_style, input_style):
    '''
    Calculates the style loss given a loss criterion
    '''
    loss = 0
    layer_weight = 1. / len(target_style)
    for i, _ in enumerate(target_style):
        # print(i)
        style_i = target_style[i].detach()
        input_i = input_style[i]
        loss_i = criterion(input_i, style_i) * layer_weight
        loss += loss_i
    return loss


def content_loss(criterion, target_content, input_content):
    '''
    Calculates the content loss given a loss criterion
    '''
    loss = 0
    for i, _ in enumerate(target_content):
        # print(i)
        content_i = target_content[i].detach()
        input_i = input_content[i]
        loss_i = criterion(input_i, content_i)
        loss += loss_i
    return loss
