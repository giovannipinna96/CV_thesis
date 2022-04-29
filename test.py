import torch
import torchvision
import os
from torch import Tensor
from torchvision.utils import make_grid
import utils
import train
from train import AverageMeter
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
    with torch.no_grad():
        for X, y in dataloader:
            X = X.to(device)
            y = y.to(device)

            y_hat = model(X)
       #     if loss_type != 'crossEntropy':
       #         y_hat = y_hat.unsqueeze(1)
            loss = loss_fn(y_hat, y) if loss_fn is not None else None
            #acc = performance(y_hat.squeeze(1), y) #TODO check if is correct
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
            step += 1

    # get final performances
    fin_loss = loss_meter.sum if loss_fn is not None else None
    fin_perf = performance_meter.avg
    print(f"TESTING - loss {fin_loss if fin_loss is not None else '--'} - performance {fin_perf:.4f}")
    
    return fin_loss, fin_perf
