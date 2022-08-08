from cv2 import threshold
from numpy import argmax
import torch
import torchvision
import os
from torch import Tensor
from torchvision.utils import make_grid
import utils
import train
from train import AverageMeter
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter  # to print to tensorboard


def test_model(model, dataloader, performance=train.accuracy, loss_fn=None, device=None, loss_type = None):
    # for tensorboard
    writer2 = SummaryWriter(f'runs/punzoni/test_model_tensorboard')
    step = 0
    # create an AverageMeter for the loss if passed
    if loss_fn is not None:
        loss_meter = AverageMeter()

    if device is None:
        device = utils.use_gpu_if_possible()

    model = model.to(device)

    performance_meter = AverageMeter()

    model.eval()
    save_values_test = []
    with torch.no_grad():
        for X, y in tqdm(dataloader):
            #X_anchor_dim = X_anchor.size(0)
            if loss_type != 'crossEntropy' and loss_type != 'triplet':
                X = torch.cat([X[0], X[1]], dim=0)
            X = X.to(device)
            y = y.to(device)
            bsz = y.shape[0]

            y_hat = model(X)
            
            if loss_type != 'crossEntropy' and loss_type != 'triplet':
                f1, f2 = torch.split(y_hat, [bsz,bsz], dim=0)
                y_hat = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
            loss = loss_fn(y_hat, y) if loss_fn is not None else None
            if loss_type != 'crossEntropy' and loss_type != 'triplet':
                acc = performance(y_hat, y.unsqueeze(-1))
            else:
                acc = performance(y_hat, y)
            if loss_fn is not None:
                loss_meter.update(loss.item(), X.shape[0])
            performance_meter.update(acc, X.shape[0])

            # stuff for tensorboard support
            img_grid = torchvision.utils.make_grid(X)
            #features = X.reshape(X.shape[0], -1)
            writer2.add_scalar('Test loss', loss_meter.avg, global_step=step)
            writer2.add_scalar(
                'Test accuracy', performance_meter.avg, global_step=step)
            writer2.add_image('Image', img_grid)
            #writer.add_embedding(features, metadata=y, lable_img= X.unsqueeze(1))
            # save loss and accurancy
            save_values_test.append(loss_meter.avg)
            save_values_test.append(performance_meter.avg)
            step += 1

    # get final performances
    fin_loss = loss_meter.sum if loss_fn is not None else None
    fin_perf = performance_meter.avg
    print(f"TESTING - loss {fin_loss if fin_loss is not None else '--'} - performance {fin_perf:.4f}")
    
    utils.save_obj(file_name="save_values_test", first=save_values_test)
    return fin_loss, fin_perf


def test_model_iiloss(model, dataloader, performance=train.accuracy, loss_fn=None, device=None, threshold = None, mean = None):
    step = 0
    # create an AverageMeter for the loss if passed
    if loss_fn is not None:
        loss_meter = AverageMeter()

    if device is None:
        device = utils.use_gpu_if_possible()

    model = model.to(device)
    performance_meter = AverageMeter()
    model.eval()
    save_values_test = []
    with torch.no_grad():
        for X, y in tqdm(dataloader):
            X = X.to(device)
            y = y.to(device)
            out_z, out_y = model(X)

            if (((mean - out_z).norm(dim=0)**2).min() >= threshold):
                y_hat = argmax(out_y)
            else:
                y_hat = -1 # not_classificable
            
            loss = loss_fn(y_hat, y) if loss_fn is not None else None
            acc = performance(y_hat, y)
            if loss_fn is not None:
                loss_meter.update(loss.item(), X.shape[0])
            performance_meter.update(acc, X.shape[0])

            # save loss and accurancy
            save_values_test.append(loss_meter.avg)
            save_values_test.append(performance_meter.avg)
            step += 1

    # get final performances
    fin_loss = loss_meter.sum if loss_fn is not None else None
    fin_perf = performance_meter.avg
    print(f"TESTING - loss {fin_loss if fin_loss is not None else '--'} - performance {fin_perf:.4f}")
    
    utils.save_obj(file_name="save_values_test", first=save_values_test)
    return fin_loss, fin_perf
