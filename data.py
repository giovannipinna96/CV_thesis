import torch
import torchvision
import numpy as np
import random
from torch.utils.data import DataLoader

class RandomPatch(torch.nn.Module):
    def __init__(self, minsize, maxsize, color_interval, prob_median:float, force_randomcolor=True):
        self.minsize=minsize
        self.maxsize=maxsize
        self.color_interval=color_interval
        self.prob_median=prob_median
        self.force_randomcolor=force_randomcolor

    def _expand_as_tensor(self, origin, destination):
        return origin.unsqueeze(-1).unsqueeze(-1).expand_as(destination)

    def __call__(self, image:torch.Tensor):
        dim = torch.randint(self.minsize, self.maxsize+1, size=(2,)).tolist()
        top_left_x = torch.randint(image.shape[1] - dim[0], size=(1,))
        top_left_y = torch.randint(image.shape[2] - dim[1], size=(1,))
        image_patch = image[:, top_left_x : top_left_x + dim[0],
              top_left_y : top_left_y + dim[1]]
        if (not self.force_randomcolor) and self.prob_median > random.random():
            image_patch[:,:,:] = self._expand_as_tensor(image.view(image.shape[0], -1).median(dim=1).values, image_patch)
        else:
            random_color = [torch.randint(low, high, size=(1,)) for low, high in zip(self.color_interval[0], self.color_interval[1])]
            random_color = torch.Tensor(random_color)
            image_patch[:,:,:] = self._expand_as_tensor(random_color, image_patch)
        return image

class PILToTensor(torch.nn.Module):
    def __call__(self, image):
        tensor = torch.from_numpy(np.array(image)).float()
        return tensor.permute(-1, 0, 1)


class To01(torch.nn.Module):
    def __call__(self, tensor:torch.Tensor):
        return tensor.sub_(tensor.min()).div_(tensor.max())

def _get_dataset(root, transform, **kwargs):
    dataset = torchvision.datasets.ImageFolder(
        root=root,
        transform=transform,
        **kwargs
    )
    return dataset

def _get_dataloader(dataset, batch_size, shuffle, num_workers, **kwargs):
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        **kwargs
    )
    return dataloader

def get_dataloaders(root_train, root_test, transform_train, transform_test, batch_size_train, batch_size_test, **kwargs):
    trainset = _get_dataset(root_train, transform_train)
    testset = _get_dataset(root_test, transform_test)

    #### TODO
    ## Aggiungere un sampler al dataloader
    trainloader = _get_dataloader(trainset, batch_size_train, shuffle=True, num_workers=8, **kwargs)
    testloader = _get_dataloader(testset, batch_size_test, shuffle=False, num_workers=8, **kwargs)
    return trainloader, testloader, trainset, testset

if __name__ == "__main__":
    ### FOR DEBUGGING LIBRARY FUNCS
    import numpy as np
    from PIL import Image
    image = Image.open("ImageSet/train/72/11_Memmi_Madonna_1000dpi__0014_Livello 15.tif")
    image = np.array(image)
    image_tensor = PILToTensor()(image)
    image_patch = RandomPatch(10, 40, [[0,0,0],[100,100,100]], .5, force_randomcolor=True)(image_tensor)
    image_orig = Image.fromarray(image_patch.permute(1,2,0).numpy().astype("uint8"))
    image_orig.save("patch.jpg")
    image_tensor_norm = To01()(image_patch)
    image_tensor_norm
