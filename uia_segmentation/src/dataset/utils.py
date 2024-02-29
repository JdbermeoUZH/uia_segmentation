import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import nibabel as nib
import nibabel.processing as nibp


def class_to_onehot(class_image, n_classes=-1, class_dim=1):
    """
    class_dim: dimension where classes are added (0 if class_image is an image and 1 if class_image is a batch)
    """
    class_image = class_image.long()
    one_hot = F.one_hot(class_image, n_classes).byte()
    one_hot = one_hot.squeeze(class_dim).movedim(-1, class_dim)
    
    return one_hot


def onehot_to_class(onehot, class_dim=1, keepdim=True):
    return onehot.argmax(dim=class_dim, keepdim=keepdim)


def normalize_quantile(data, min_p=0, max_p=1.0):
    min = torch.quantile(data, min_p)
    max = torch.quantile(data, max_p)
    return normalize_min_max(data, min, max)

def normalize_min_max(data, min=None, max=None, scale: float = 1):
    if min is None:
        min = torch.min(data)
    if max is None:
        max = torch.max(data)

    if max == min:
        data = torch.zeros_like(data)
    else:
        data = (data - min) / (max - min)
    data = torch.clip(data, 0, 1)
    data = scale * data
    
    if scale == 255:
        data = data.to(torch.uint8)
    
    return data

def normalize(type: str, **kwargs):
    if type == 'min_max':
        return normalize_min_max(**kwargs)
    elif type == 'quantile':
        return normalize_quantile(**kwargs)
    else:
        raise ValueError(f'Unknown normalization type: {type}')


def resize_and_resample_nibp(
        image: np.ndarray,
        target_size: tuple[int, ...],
        original_voxel_size: tuple[float, ...],
        target_voxel_size: tuple[float, ...],
        order: int
) -> np.ndarray:
    """
    Resizes and resamples a 3D image to a given voxel size and target size.
    """
    # Convert image to nibabel image
    affine_matrix = np.eye(4)
    affine_matrix[0, 0] = original_voxel_size[0]
    affine_matrix[1, 1] = original_voxel_size[1]
    affine_matrix[2, 2] = original_voxel_size[2]

    # If the image is 3D with 3 dim array, just resample it
    if len(image.shape) == 3:
        return nibp.conform(
            nib.Nifti1Image(image, affine_matrix), target_size, target_voxel_size, order=order
        ).get_fdata()

    # If the image has multiple channels, resample each channel
    new_img_channels = []
    for channel_i in range(image.shape[0]):
        nib_img = nib.Nifti1Image(image[channel_i], affine_matrix)
        new_img_channels.append(
            nibp.conform(nib_img, target_size, target_voxel_size, order=order).get_fdata()
        )

    return np.stack(new_img_channels, axis=0)


class RBF(nn.Module):
    """
    Alternative to ReLU activation function

    """

    def __init__(self, n_channels, mean=0.2, stddev=0.05, n_dimensions=2):
        super().__init__()

        self.mean = mean
        self.stddev = stddev

        image_shape = [1] * n_dimensions

        self.scale = nn.Parameter(torch.empty((1, n_channels, *image_shape)), requires_grad=True)
        self.reset_parameters()

    def reset_parameters(self):
        with torch.no_grad():
            self.scale.normal_(self.mean, self.stddev)

    def forward(self, x):
        y = torch.exp(-(x ** 2) / (self.scale ** 2))
        return y
