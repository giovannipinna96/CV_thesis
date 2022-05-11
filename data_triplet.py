import torch
from torch.utils.data.dataloader import DataLoader
import torchvision
from torchvision import transforms as T
import random
import time
from functools import reduce
from PIL import Image

from data import To01

class TripletDataset(torchvision.datasets.ImageFolder):
    def __init__(self, root, transform, lazy:bool=True, img_dim:int=256, train:bool=True, **kwargs):
        super().__init__(root, transform, **kwargs)
        self.lazy = lazy
        self.train = train
        if not lazy:
            self.imgs = [T.Resize((img_dim, img_dim))(T.functional.pil_to_tensor(Image.open(img_path))) for img_path, _ in self.imgs]
            self.imgs = torch.stack(self.imgs)
        self.indices_per_class = {}
        self.labels = list(range(len(self.classes)))
        for cl in self.labels:
            self.indices_per_class[cl] = [i for i, target in enumerate(self.targets) if target == cl]
        # self.timer = {
        #     "anchor_label": [],
        #     "positive_idx": [],
        #     "negative_idx": [],
        #     "getitem": [],
        # }


    def __getitem__(self, index: int):
        if self.train:
            # tic = time.perf_counter()
            anchor_label = self.targets[index]
            # toc1 = time.perf_counter()
            positive_idx = random.choice(list(set(self.indices_per_class[anchor_label]).difference({index})))
            # toc2 = time.perf_counter()
            negative_idx = random.choice(reduce(lambda l1, l2: l1 + l2, [self.indices_per_class[label] for label in self.labels if label != anchor_label]))
            # toc3 = time.perf_counter()
            if self.lazy:
                anc, pos, neg, lab = super().__getitem__(index)[0], super().__getitem__(positive_idx)[0], super().__getitem__(negative_idx)[0], anchor_label
            else:
                anc = self.imgs[index]
                pos = self.imgs[positive_idx]
                neg = self.imgs[negative_idx]
                lab = anchor_label
                if self.transform is not None:
                    anc = self.transform(anc)
                    pos = self.transform(pos)
                    neg = self.transform(neg)
            # toc4 = time.perf_counter()
            # self.timer["anchor_label"].append(toc1-tic)
            # self.timer["positive_idx"].append(toc2-toc1)
            # self.timer["negative_idx"].append(toc3-toc2)
            # self.timer["getitem"].append(toc4-toc3)
            return anc, pos, neg, lab
        if self.lazy:
            return super().__getitem__(index)
        img = self.imgs[index]
        lab = self.targets[index]
        if self.transform is not None:
            img = self.transform(img)
        return img, lab

def get_dataloaders(root_train, root_test, batch_size_train, batch_size_test, transform_train=None, transform_test=None, lazy=True, img_dim=256, **kwargs):
    trainset = TripletDataset(root_train, transform_train, lazy=lazy, img_dim=img_dim, train=True)
    testset = TripletDataset(root_test, transform_test, lazy=lazy, img_dim=img_dim, train=False)
    trainloader = DataLoader(trainset, batch_size_train, shuffle=True, **kwargs)
    testloader = DataLoader(testset, batch_size_test, shuffle=False, **kwargs)
    return trainloader, testloader

if __name__ == "__main__":
    transform_test = T.Compose([
        T.Resize((256, 256)),
        To01(),
        T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])
    dataset = TripletDataset("ImageSet/AllData", transform_test, lazy=False)
    # a,b,c,d = dataset[0]
    dataloader = torch.utils.data.DataLoader(dataset, 128, shuffle=False, num_workers=0)
    dl = next(iter(dataloader))
    dataset