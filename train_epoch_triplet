import imp
import torch

def train_epoch_triplet(model, dataloader, loss_triplet_fn, optimizer, loss_meter, device, lr_scheduler, optim_step_each_ite=1):
    for i, (X_anchor, X_pos, X_neg, y) in enumerate(dataloader):
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
        # valutare se aggiungere qualcosa qui, potrebbe non aver senso calcolare l'accuracy in fase di train
        # 7. update the loss and accuracy AverageMeter
        loss_meter.update(val=loss.item(), n=X.shape[0])
        # performance_meter.update(val=acc, n=X.shape[0])