from torch.nn import ModuleList
from torchvision import transforms as T
import data

def get_transform_train():
    transform_train = T.Compose([
        T.Resize((256, 256)),
        T.RandomRotation((0, 359), interpolation=T.InterpolationMode.BILINEAR),
        T.RandomApply(
            ModuleList([T.GaussianBlur(kernel_size=5)]),
            p=.33),
        data.PILToTensor(),
        T.RandomApply(
            ModuleList([data.RandomPatch(50, 200, [[0,0,0],[100,100,100]], .5)]),
            p=0.75),
        data.To01(),
        T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])

    return transform_train

def get_transform_test():
    transform_test = T.Compose([
        T.Resize((256, 256)),
        T.ToTensor(),
        T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])

    return transform_test