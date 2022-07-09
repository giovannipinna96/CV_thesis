from torch.nn import ModuleList
from torchvision import transforms as T
import data


def get_transform_train():
    """This function do some random transformation for aumented the dataset (only for train).
    In particular:
    1. resize the img in 256x256
    2. random rotation (0,359) with bilinear interpolation
    3. random gaussian blur (kernel_size = 5, probability = 0.3)
    4. random patch
    5. transform to tensor
    6. tensor normalization

    Returns:
        Torch.tensor: returns the tensor that represents the image after all the transformations
    """
    transform_train = T.Compose([
        T.Resize((256, 256)),
        T.RandomRotation((0, 359), interpolation=T.InterpolationMode.BILINEAR),
        T.RandomApply(
            ModuleList([T.GaussianBlur(kernel_size=5)]),
            p=.33),
        data.PILToTensor(),
        T.RandomApply(
            ModuleList(
                [data.RandomPatch(50, 200, [[0, 0, 0], [100, 100, 100]], .5)]),
            p=0.75),
        data.To01(),
        T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])

    return transform_train


def get_transform_test():
    """This function do some random transformation for aumented the dataset (only for test).
    We don't care in the case of the head to apply patches or blurs. In particular:
    1. resize 256x256
    2. transform to tensor
    3. tensor normalization
    Returns:
        Torch.tensor: returns the tensor that represents the image after all the transformations
    """
    transform_test = T.Compose([
        T.Resize((256, 256)),
        T.ToTensor(),
        T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])

    return transform_test
