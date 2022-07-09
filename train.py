from cProfile import label
import imp
from cv2 import norm
from numpy import diff
import torch
import torchvision
import os
from torch import Tensor
from torchvision.utils import make_grid
import utils
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter  # to print to tensorboard


def accuracy(nn_output: Tensor, ground_truth: Tensor, k: int = 1):
    '''
    Return accuracy@k for the given model output and ground truth
    nn_output: a tensor of shape (num_datapoints x num_classes) which may 
       or may not be the output of a softmax or logsoftmax layer
    ground_truth: a tensor of longs or ints of shape (num_datapoints)
    k: the 'k' in accuracy@k
    '''
    assert k <= nn_output.shape[1], f"k too big. Found: {k}. Max: {nn_output.shape[1]} inferred from the nn_output"
    # get classes of assignment for the top-k nn_outputs row-wise
    nn_out_classes = nn_output.topk(k).indices
    # make ground_truth a column vector
    ground_truth_vec = ground_truth.unsqueeze(-1)
    # and repeat the column k times (= reproduce nn_out_classes shape)
    ground_truth_vec = ground_truth_vec.expand_as(nn_out_classes)
    # produce tensor of booleans - at which position of the nn output is the correct class located?
    correct_items = (nn_out_classes == ground_truth_vec)
    # now getting the accuracy is easy, we just operate the sum of the tensor and divide it by the number of examples
    acc = correct_items.sum().item() / nn_output.shape[0]
    return acc


class AverageMeter(object):
    '''
    a generic class to keep track of performance metrics during training or testing of models
    '''

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def train_epoch_triplet(model, dataloader, loss_triplet_fn, optimizer, loss_meter,
                        device, lr_scheduler, performance_meter, performance,  optim_step_each_ite=1):
    writer = SummaryWriter(f'runs/punzoni/tryout_ternsorboard')
    step = 0
    save_values = []
    for i, (X_anchor, X_pos, X_neg, y) in tqdm(enumerate(dataloader)):
        X_anchor = X_anchor.to(device)
        X_anchor_dim = X_anchor.size(0)
        y = y.to(device)
        X_pos = X_pos.to(device)
        X_neg = X_neg.to(device)
        X = torch.cat((X_anchor, X_pos, X_neg))
        # 1. reset the gradients previously accumulated by the optimizer
        #    this will avoid re-using gradients from previous loops
        optim_step_this_ite = ((i+1) % optim_step_each_ite) == 0
        if optim_step_this_ite:
            optimizer.zero_grad() 
        # 2. get the predictions from the current state of the model
        #    this is the forward pass
        y_hat = model(X)
        # 3. calculate the loss on the current mini-batch
        loss = loss_triplet_fn(y_hat[:X_anchor_dim], y_hat[X_anchor_dim:X_anchor_dim*2], y_hat[X_anchor_dim*2:])
        # 4. execute the backward pass given the current loss
        loss.backward()
        # 5. update the value of the params
        if optim_step_this_ite:
            optimizer.step()
            if lr_scheduler is not None:
                lr_scheduler.step()
        # 6. calculate the accuracy for this mini-batch
        acc = performance(y_hat[:X_anchor_dim], y)
        # valutare se aggiungere qualcosa qui, potrebbe non aver senso calcolare l'accuracy in fase di train
        # 7. update the loss and accuracy AverageMeter
        loss_meter.update(val=loss.item(), n=X.shape[0])
        performance_meter.update(val=acc, n=X.shape[0])
        # performance_meter.update(val=acc, n=X.shape[0])
        # stuff for tensorboard support
        img_grid = torchvision.utils.make_grid(X)
        #features = X.reshape(X.shape[0], -1)
        writer.add_scalar('Training loss', loss_meter.avg, global_step=step)
        writer.add_scalar('Training accuracy',
                          performance_meter.avg, global_step=step)
        writer.add_image('Image', img_grid)
        #writer.add_embedding(features, metadata=y, lable_img= X.unsqueeze(1))
        # save loss and accurancy
        save_values.append(loss_meter.avg)
        save_values.append(performance_meter.avg)

        print(f" for batch {i} we have loss avg = {loss_meter.avg}")
        print(f"for batch {i} we have performancemeter avg {performance_meter.avg}")

        step += 1
    
    return save_values

# note: I've added a generic performance to replace accuracy
def train_epoch(
    model, dataloader, loss_fn, optimizer, loss_meter, performance_meter, performance, device,
    lr_scheduler, loss_type
):
    # support for tensorboard
    writer = SummaryWriter(f'runs/punzoni/tryout_ternsorboard')
    step = 0
    save_values = []
    for X, y in tqdm(dataloader):
        if loss_type != 'crossEntropy':
            X = torch.cat([X[0], X[1]], dim=0)
        X = X.to(device)
        y = y.to(device)
        bsz = y.shape[0]
        # 1. reset the gradients previously accumulated by the optimizer
        #    this will avoid re-using gradients from previous loops
        optimizer.zero_grad()
        # 2. get the predictions from the current state of the model
        #    this is the forward pass
        y_hat = model(X)
        # 3. calculate the loss on the current mini-batch
        if loss_type != 'crossEntropy':
            f1, f2 = torch.split(y_hat, [bsz,bsz], dim=0)
            y_hat = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
        loss = loss_fn(y_hat, y) 
        # 4. execute the backward pass given the current loss
        loss.backward()
        # 5. update the value of the params
        optimizer.step()
        if lr_scheduler is not None:
            lr_scheduler.step()
        # 6. calculate the accuracy for this mini-batch
        if loss_type != 'crossEntropy':
            acc = performance(y_hat, y.unsqueeze(-1))
        else:
            acc = performance(y_hat, y)
        # 7. update the loss and accuracy AverageMeter
        loss_meter.update(val=loss.item(), n=X.shape[0])
        performance_meter.update(val=acc, n=X.shape[0])

        # stuff for tensorboard support
        img_grid = torchvision.utils.make_grid(X)
        #features = X.reshape(X.shape[0], -1)
        writer.add_scalar('Training loss', loss_meter.avg, global_step=step)
        writer.add_scalar('Training accuracy',
                          performance_meter.avg, global_step=step)
        writer.add_image('Image', img_grid)
        #writer.add_embedding(features, metadata=y, lable_img= X.unsqueeze(1))
        # save loss and accurancy
        save_values.append(loss_meter.avg)
        save_values.append(performance_meter.avg)
        step += 1
    
    return save_values

def train_epoch_iiloss(
    model, dataloader, optimizer, ii_loss_meter, ii_performance_meter,ce_loss_meter, ce_performance_meter, performance, device,
    lr_scheduler, num_classes
):
    loss_fn = torch.nn.CrossEntropyLoss()
    step = 0
    ii_save_values = []
    ce_save_values = []
    for X, y in tqdm(dataloader):
        X = X.to(device)
        y = y.to(device)
        # 1. reset the gradients previously accumulated by the optimizer
        #    this will avoid re-using gradients from previous loops
        optimizer.zero_grad()
        # 2. get the predictions from the current state of the model
        #    this is the forward pass
        out_z, out_y = model(X)
        # 3. calculate the iiloss on the current mini-batch
        ii_loss = compute_ii_loss(out_z, y, num_classes) 
        # 4. execute the backward pass given the current loss
        ii_loss.backward(retain_graph = True)
        # 5. calculate the iiloss on the current mini-batch
        ce_loss = loss_fn(out_y, y)
        # 6. execute the backward pass given the current loss
        ce_loss.backward()
        # 7. update the value of the params
        optimizer.step()
        if lr_scheduler is not None:
            lr_scheduler.step()
        # 8. calculate the accuracy for this mini-batch
        ii_acc = performance(out_z, y)
        ce_acc = performance(out_y, y)
        # 9. update the loss and accuracy AverageMeter
        ii_loss_meter.update(val=ii_loss.item(), n=X.shape[0])
        ii_performance_meter.update(val=ii_acc, n=X.shape[0])
        ce_loss_meter.update(val=ce_loss.item(), n=X.shape[0])
        ce_performance_meter.update(val=ce_acc, n=X.shape[0])

        #writer.add_embedding(features, metadata=y, lable_img= X.unsqueeze(1))
        # save loss and accurancy
        ii_save_values.append(ii_loss_meter.avg)
        ii_save_values.append(ii_performance_meter.avg)
        ce_save_values.append(ce_loss_meter.avg)
        ce_save_values.append(ce_performance_meter.avg)
        step += 1
    
    return ii_save_values, ce_save_values


def compute_embeddings(model, dataloader, device):
    embeddings = []
    labels = []
    model.eval()
    for X, y in tqdm(dataloader):
        X = X.to(device)
        y = y.to(device)
        labels.append(y)
        out_z, _ = model(X)
        embeddings.append(out_z)

    embedding = torch.stack(embeddings)
    label = torch.stack(labels)
    mean = bucket_mean(embedding, label)

    #TODO c'è un return? quale?

def train_model(
    model, dataloader, loss_fn, optimizer, num_epochs, checkpoint_loc=None, checkpoint_name="checkpoint.pt",
    performance=accuracy, lr_scheduler=None, device=None, lr_scheduler_step_on_epoch=True, loss_type='crossEntropy'
):

    # create the folder for the checkpoints (if it's not None)
    if checkpoint_loc is not None:
        os.makedirs(checkpoint_loc, exist_ok=True)

    if device is None:
        device = utils.use_gpu_if_possible()

    model = model.to(device)
    model.train()

    # epoch loop
    save_values_train = []
    for epoch in range(num_epochs):

        loss_meter = AverageMeter()
        performance_meter = AverageMeter()

        # added print for LR
        print(
            f"Epoch {epoch+1} --- learning rate {optimizer.param_groups[0]['lr']:.5f}")

        lr_scheduler_batch = lr_scheduler if not lr_scheduler_step_on_epoch else None
        if type(loss_fn) == torch.nn.modules.loss.TripletMarginLoss:
            v = train_epoch_triplet(model, dataloader, loss_fn, optimizer, loss_meter, device, lr_scheduler_batch, performance_meter,
              performance, optim_step_each_ite=1)
        
        elif type(loss_fn) == torch.nn.modules.loss.CrossEntropyLoss:
            v = train_epoch(model, dataloader, loss_fn, optimizer, loss_meter, performance_meter,
                        performance, device, lr_scheduler_batch, loss_type)

        else:
            ii_loss_meter = AverageMeter()
            ii_performance_meter = AverageMeter()
            ce_loss_meter = AverageMeter()
            ce_performance_meter = AverageMeter()
            ii, ce = train_epoch_iiloss(model, dataloader, optimizer, ii_loss_meter, ii_performance_meter, ce_loss_meter, ce_performance_meter,
                        performance, device, lr_scheduler_batch, loss_type)
        
        save_values_train.append(v)
        #save_values_train.append(v2)

        print(f"Epoch {epoch+1} completed. Loss - total: {loss_meter.sum:.4f} - average: {loss_meter.avg:.4f}; Performance: {performance_meter.avg:.4f}")

        # produce checkpoint dictionary -- but only if the name and folder of the checkpoint are not None
        if checkpoint_name is not None and checkpoint_loc is not None:
            checkpoint_dict = {
                "parameters": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "loss sum" : loss_meter.sum,
                "loss avg" : loss_meter.avg,
                "preformance_meter" : performance_meter.avg
            }
            torch.save(checkpoint_dict, os.path.join(checkpoint_loc, checkpoint_name))

        if lr_scheduler is not None and lr_scheduler_step_on_epoch:
            # Reduce learning rate when a metric has stopped improving. 
            if isinstance(lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                # Note that step should be called after the validation that is (loss_meter.avg)
                lr_scheduler.step(loss_meter.avg)
            else:
                lr_scheduler.step()
    utils.save_obj(file_name="save_value_train", first= save_values_train)
    return loss_meter.sum, performance_meter.avg



def compute_ii_loss(out_z, labels, num_classes):
    intra_spread = torch.Tensor([0])
    inter_separation = torch.inf
    class_mean = bucket_mean(out_z, labels, num_classes)
    for j in range(num_classes):
        data_class = out_z[labels == j]
        difference_from_mean = data_class - class_mean[j].unsqueeze(1)
        norm_from_mean = difference_from_mean.norm()**2
        intra_spread += norm_from_mean
        class_mean_previous = class_mean[:j]
        norm_form_previous_means = (class_mean_previous - class_mean[j]).norm()**2
        inter_separation = min(inter_separation, norm_form_previous_means.min())

    return intra_spread, inter_separation

def bucket_mean(embeddings, labels, num_classes):
    tot = torch.zeros(num_classes).scatter_add(0, labels, embeddings)
    count = torch.zeros(num_classes).scatter_add(0, labels, torch.ones_like(embeddings))

    return tot/count