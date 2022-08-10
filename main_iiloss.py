import argparse
import torch
import transformation
import os
import data
import train
import test
from allParameters import allParameters
import createNet
from numpy import percentile
import torch
import os
from torch import Tensor
import utils
from tqdm import tqdm
from numpy import argmax

def not_freeze(net, layers: list):
    """Defrost layers.

    Args:
        net (torchvision.models): The network.
        layers (list): list of layers to defrost

    Returns:
        torchvision.models: The network with defrost layer in list layers.
    """
    for _, param in net.named_parameters():
        param.requires_grad = False

    # freeze layers
    for name, param in net.named_parameters():
        if name in layers:
            param.requires_grad = True

    return net

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

def train_model(
    model, dataloader, loss_fn, optimizer, num_epochs, checkpoint_loc=None, checkpoint_name="checkpoint.pt",
    performance=accuracy, lr_scheduler=None, device=None, lr_scheduler_step_on_epoch=True, loss_type='crossEntropy', num_classes=None
):
    threshold = None
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

        ii_loss_meter = AverageMeter()
        ii_performance_meter = AverageMeter()
        ce_loss_meter = AverageMeter()
        ce_performance_meter = AverageMeter()
        ii, ce = train_epoch_iiloss(model, dataloader, loss_fn, optimizer, ii_loss_meter, ii_performance_meter, ce_loss_meter, ce_performance_meter,
                        performance, device, lr_scheduler_batch, num_classes=num_classes)
        
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
    
    print('Compute threshold')
    threshold, mean = compute_threshold(model, dataloader, num_classes, device)


    if threshold is None:
        return loss_meter.sum, performance_meter.avg
    else:
        return loss_meter.sum, performance_meter.avg, threshold, mean

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

def train_epoch_iiloss(
    model, dataloader, loss_fn, optimizer, ii_loss_meter, ii_performance_meter,ce_loss_meter, ce_performance_meter, performance, device,
    lr_scheduler, num_classes
):
    print('Start Train ii loss')
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
        if not (step % 2):
            ii_loss.backward(retain_graph = True)
        # 5. calculate the iiloss on the current mini-batch
        #if step % 2 :
        #    ce_loss = loss_fn(out_y, y)
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


def compute_embeddings(model, dataloader, num_classes, device):
    embeddings = []
    labels = []
    model.eval()
    with torch.no_grad():
        for X, y in tqdm(dataloader):
            X = X.to(device)
            y = y.to(device)
            out_z, _ = model(X)
            labels.append(y)
            embeddings.append(out_z)

    embedding = torch.cat(embeddings)
    label = torch.cat(labels)
    mean = bucket_mean(embedding, label, num_classes)

    return embedding, label, mean  


def compute_threshold(model, dataloder, num_classes, device):
    embedding, label, mean = compute_embeddings(model, dataloder, num_classes, device)
    outlayer_score = []
    for j in range(embedding.shape[0]):
        outlayer_score.append(((mean - embedding[j]).norm(dim=0)**2).min()) 
    outlayer_score.sort()
    threshold = percentile(outlayer_score, 1)
    
    return threshold, mean

def compute_ii_loss(out_z, labels, num_classes):
    intra_spread = torch.Tensor([0]).cuda()
    inter_separation = torch.tensor(float('inf')).cuda()# torch.inf.cuda()
    class_mean = bucket_mean(out_z, labels, num_classes) 
    for j in range(num_classes):
        data_class = out_z[labels == j]
        difference_from_mean = data_class - class_mean[j]
        norm_from_mean = difference_from_mean.norm()**2
        intra_spread += norm_from_mean
        class_mean_previous = class_mean[:j]
        norm_form_previous_means = (class_mean_previous - class_mean[j]).norm()**2
        inter_separation = min(inter_separation, norm_form_previous_means.min())

    return intra_spread - inter_separation

def bucket_mean(embeddings, labels, num_classes):
    tot = torch.zeros(num_classes, embeddings.shape[1], device=torch.device('cuda')).index_add(0, labels, embeddings)
    count = torch.zeros(num_classes, embeddings.shape[1], device=torch.device('cuda')).index_add(0, labels, torch.ones_like(embeddings))

    return tot/count

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
            y_hat = []
            for j in range(out_z.shape[0]):
                if (((mean - out_z[j]).norm(dim=0)**2).min() >= threshold):
                    y_hat.append(argmax(out_y[j].cpu()))
                else:
                    y_hat.append(torch.tensor(-1)) # not_classificable
            
            loss = loss_fn(out_y, y) if loss_fn is not None else None
            acc = performance(out_y, y)
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--root_train", type=str, default="ImageSet/train")
    parser.add_argument("--root_test", type=str, default="ImageSet/test")
    parser.add_argument("--loss_type", type=str, default="iiloss")
    parser.add_argument("--optimizer", type=str, default="sgd")
    parser.add_argument("--out_net", type=int, default=18)
    parser.add_argument("--is_feature_extraction", type=bool, default=True)
    parser.add_argument("--weights_save_path", type=str, default="models/model_alte.pt")
    parser.add_argument("--pickle_save_path", type=str, default="out_ii")
    parser.add_argument("--is_ml", type=bool, default=True)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument
    args = parser.parse_args()
    # set all parameters
    allParams = allParameters(
        root_train=args.root_train,
        root_test=args.root_test,
        weights_save_path=args.weights_save_path,
        batch_size_train=32,
        batch_size_test=128,
        model='resnet50',
        pretrained=True,
        num_epochs=args.epochs,
        not_freeze='nothing',
        loss_type=args.loss_type,
        out_net=args.out_net,
        is_feature_extraction=args.is_feature_extraction,
        is_ml = args.is_ml,
        optimizer=args.optimizer
    )

    # transform the dataset
    transform_train = transformation.get_transform_train()
    transform_test = transformation.get_transform_test()

    trainloader, testloader, trainset, testset = data.get_dataloaders(allParams.get_root_train(),
                                                                    allParams.get_root_test(),
                                                                    transform_train,
                                                                    transform_test,
                                                                    allParams.get_batch_size_train(),
                                                                    allParams.get_batch_size_test(),
                                                                    balance=True
                                                                    ) 

    #define the number of different classes
    num_classes = len(trainset.classes)
    loss_fn = torch.nn.CrossEntropyLoss()
    net = createNet.resNet50Costum(num_classes)
    dict_custom_resnet50, classic = createNet.create_dict_resNet50Costum(net, "resnet50_aug_per_giovanni.pt_resnet50.pt")
    net.load_state_dict(dict_custom_resnet50)
    #net = not_freeze(net, ['layer4.0.conv1.weight', 'layer4.0.bn1.weight', 'layer4.0.bn1.bias',
    #'layer4.0.conv2.weight', 'layer4.0.bn2.weight', 'layer4.0.bn2.bias', 'layer4.0.conv3.weight',
    #'layer4.0.bn3.weight', 'layer4.0.bn3.bias', 'layer4.0.downsample.0.weight',
    #'layer4.0.downsample.1.weight', 'layer4.0.downsample.1.bias', 'layer4.1.conv1.weight',
    #'layer4.1.bn1.weight', 'layer4.1.bn1.bias', 'layer4.1.conv2.weight', 'layer4.1.bn2.weight',
    #'layer4.1.bn2.bias', 'layer4.1.conv3.weight', 'layer4.1.bn3.weight', 'layer4.1.bn3.bias',
    #'layer4.2.conv1.weight', 'layer4.2.bn1.weight', 'layer4.2.bn1.bias', 'layer4.2.conv2.weight',
    #'layer4.2.bn2.weight', 'layer4.2.bn2.bias', 'layer4.2.conv3.weight', 'layer4.2.bn3.weight',
    #'layer4.2.bn3.bias', 'fc1.weight', 'fc1.bias', 'fc2.weight', 'fc2.bias'])
    if allParams.optimizer.lower() == "sgd":
        optimizer = torch.optim.SGD(net.parameters(),
                                    lr=.0001,
                                    momentum=.9,
                                    weight_decay=5e-4
                                    )
    elif allParams.optimizer.lower() == "radam":
        optimizer = torch.optim.RAdam(net.parameters(), lr=.0001)
    else:
        raise NotImplementedError(f"Invalid optimizer {allParams.optimizer}. Please choose from 'sgd' or 'radam'.")
    
    # set scheduler
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                     milestones=[50,75],
                                                     gamma=0.1
                                                     )

    # train
    print('Start Train')
    _, _, threshold, mean = train_model(net,
                      trainloader,
                      loss_fn,
                      optimizer,
                      allParams.get_num_epochs(),
                      lr_scheduler=scheduler,
                      device=allParams.get_device(),
                      loss_type=allParams.get_loss_type(),
                      num_classes=num_classes
                      )

    print('Start Test ii loss')
    test_model_iiloss(net,
                        testloader,
                        loss_fn=loss_fn,
                        device=allParams.get_device(),
                        threshold=threshold,
                        mean = mean
                        )

    print('Saving weights...')
    os.makedirs(os.path.dirname(allParams.get_weights_save_path()),
                exist_ok=True
                )
    torch.save(net.state_dict(), allParams.get_weights_save_path())
    
    print('Saving pickle')
    utils.save_obj(file_name=f"./pickle_thres_mean_alte",
                        first=threshold,
                        second=mean
                        )

print('Finish')