import torch
import torchvision
import random
from torch.utils.data import DataLoader

class RandomPatch():
    def __init__(self, minsize, maxsize, color_interval, prob_median:float):
        self.minsize=minsize
        self.maxsize=maxsize
        self.color_interval=color_interval
        self.prob_median=prob_median


    def __call__(self, image:torch.Tensor):
        dim = torch.randint(self.minsize, self.maxsize+1, size=(2,))
        top_left_x = torch.randint(image.shape[0] - dim[0])
        top_left_y = torch.randint(image.shape[1] - dim[1])
        image[top_left_x : top_left_x + dim[0],
              top_left_y : top_left_y + dim[1]] = image.median() if random.random() < self.prob_median else torch.randint(self.color_interval[0], self.color_interval[1]+1)
        return image

class PILToTensor():
    def __call__(self, image):
        return torch.Tensor(image)

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
    trainset = _get_dataset("ImageSet/train", transform=None)
    testset = _get_dataset("ImageSet/test", transform=None)