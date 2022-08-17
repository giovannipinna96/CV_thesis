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
import numpy as np



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
        train_epoch_iiloss(model, dataloader, loss_fn, optimizer, ii_loss_meter, ii_performance_meter, ce_loss_meter, ce_performance_meter,
                        performance, device, lr_scheduler_batch, num_classes=num_classes)
        
        print(f"Epoch {epoch+1} completed. Loss - total: II:{ii_loss_meter.avg:.4f}; CE:{ce_loss_meter.avg:.4f} - Performance: {ce_performance_meter.avg:.4f}")

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


    #if threshold is None:
    return loss_meter.sum, performance_meter.avg
    #else:
    #    return loss_meter.sum, performance_meter.avg, threshold, mean

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
    for i, (X, y) in enumerate(tqdm(dataloader)):
        X = X.to(device)
        y = y.to(device)
        # 1. reset the gradients previously accumulated by the optimizer
        #    this will avoid re-using gradients from previous loops
        optimizer.zero_grad()
        # 2. get the predictions from the current state of the model
        #    this is the forward pass
        out_z, out_y = model(X)
        # 3. calculate the iiloss on the current mini-batch
        if (i % 2 == 0) :
            ii_loss = compute_ii_loss(out_z, y, num_classes)# * 0.01
        # 4. execute the backward pass given the current loss
            ii_loss.backward() #retain_graph = True
        # 5. calculate the iiloss on the current mini-batch
        if (i % 2 == 1) :
            ce_loss = loss_fn(out_y, y)
        # 6. execute the backward pass given the current loss
            ce_loss.backward()
        # 7. update the value of the params
        optimizer.step()
        if lr_scheduler is not None:
            lr_scheduler.step()
        # 8. calculate the accuracy for this mini-batch
        if (i % 2 == 0) :
            ii_acc = performance(out_z, y)
            ii_loss_meter.update(val=ii_loss.item(), n=X.shape[0])
            ii_performance_meter.update(val=ii_acc, n=X.shape[0])
        if (i % 2 == 1):
            ce_acc = performance(out_y, y)
            ce_loss_meter.update(val=ce_loss.item(), n=X.shape[0])
            ce_performance_meter.update(val=ce_acc, n=X.shape[0])

        #writer.add_embedding(features, metadata=y, lable_img= X.unsqueeze(1))
        # save loss and accurancy
        if (i % 2 == 0):
            ii_save_values.append(ii_loss_meter.avg)
            ii_save_values.append(ii_performance_meter.avg)
        if (i % 2 == 1):
            ce_save_values.append(ce_loss_meter.avg)
            ce_save_values.append(ce_performance_meter.avg)
        step += 1
    
    #return 0,0 #ii_save_values, ce_save_values

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
    mean = bucket_mean(embedding, label, num_classes=label.max().item()+1)

    return embedding, label, mean  

def compute_threshold(model, dataloder, num_classes, device):
    embedding, label, mean = compute_embeddings(model, dataloder, num_classes, device)
    outlier_score = []
    #for j in range(embedding.shape[0]):
    #    outlier_score.append(((mean - embedding[j]).norm(dim=1)**2).min()) 
    outlier_score_val = outlier_score(embedding, mean)
    outlier_score_val2 = outlier_score_val.tolist()
    outlier_score.sort()
    threshold = percentile(outlier_score, 1)
    
    return threshold, mean

def compute_ii_loss(out_z, labels, num_classes):
    n_datapoints = len(out_z)
    device = out_z.device
    delta = 0.5
    intra_spread = torch.Tensor([0]).to(device)
    inter_separation = torch.Tensor([float("inf")]).to(device)
    class_mean = bucket_mean(out_z, labels, num_classes)
    empty_classes = []

    for j in range(num_classes):
        # update intra_spread
        data_class = out_z[labels == j]
        if len(data_class) == 0:
            empty_classes.append(j)
            continue
        difference_from_mean = data_class - class_mean[j]
        norm_from_mean = difference_from_mean.norm()**2
        intra_spread += norm_from_mean
        # update inter_separation
        class_mean_previous = class_mean[list(set(range(j)).difference(empty_classes))]
        if class_mean_previous.shape[0] > 0:
            norm_from_previous_means = (class_mean_previous - class_mean[j]).norm(dim=1)**2
            inter_separation = min(inter_separation, norm_from_previous_means.min())
        
    return intra_spread/n_datapoints - min(delta, inter_separation)

def bucket_mean(embeddings, labels, num_classes):
    device = embeddings.device
    tot = torch.zeros(num_classes, embeddings.shape[1], device=device).index_add(0, labels, embeddings)
    count = torch.zeros(num_classes, embeddings.shape[1], device=device).index_add(0, labels, torch.ones_like(embeddings))

    return tot/count

def test_model_iiloss(model, dataloader, performance=train.accuracy, loss_fn=None, device=None,
                        threshold = None, mean = None):
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
    m = []
    with torch.no_grad():
        for X, y in tqdm(dataloader):
            X = X.to(device)
            y = y.to(device)
            out_z, out_y = model(X)
            y_hat = []
            for j in range(out_z.shape[0]):
                if (((mean - out_z[j]).norm(dim=1)**2).min() >= threshold):
                    m.append(((mean - out_z[j]).norm(dim=1)**2))
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
    mm = torch.stack(m)
    print (mm.max())
    print (mm.min())
    print (mm.mean())
    print(f"TESTING - loss {fin_loss if fin_loss is not None else '--'} - performance {fin_perf:.4f}")
    
    utils.save_obj(file_name="save_values_test", first=save_values_test)
    return fin_loss, fin_perf

def test_model_on_extra(model, dataloader, device=None, threshold = None, mean = None):
    step = 0
    if device is None:
        device = utils.use_gpu_if_possible()

    model = model.to(device)
    y_hat = []
    m = []
    model.eval()
    with torch.no_grad():
        for X, y in tqdm(dataloader):
            X = X.to(device)
            y = y.to(device)
            out_z, out_y = model(X)
            for j in range(out_z.shape[0]):
                if (((mean - out_z[j]).norm(dim=1)**2).min() >= threshold):
                    m.append(((mean - out_z[j]).norm(dim=1)**2))
                    y_hat.append(0)
                else:
                    y_hat.append(1) # not_classificable
            step += 1
    mm = torch.stack(m)
    print (mm.max())
    print (mm.min())
    print (mm.mean())
    print(f"TESTING on EXTRA - performance {np.mean(y_hat):.4f}")

def eval_outlier_scores(dataloader:torch.utils.data.DataLoader, model:torch.nn.Module, traindata_means:torch.Tensor, device:torch.device) -> torch.Tensor:
    '''
    Evaluates the outlier scores for a model on a dataloader.
    '''
    model.to(device)
    model.eval()
    with torch.no_grad():
        outlier_scores = torch.zeros(len(dataloader.dataset))
        for i, (X, y) in enumerate(tqdm(dataloader)):
            X = X.to(device)
            y = y.to(device)
            embeddings, y_hat = model(X)
            outlier_scores_batch = outlier_score(embeddings, traindata_means)
            outlier_scores[i*X.shape[0]:(i+1)*X.shape[0]] = outlier_scores_batch
    return outlier_scores

def outlier_score(embeddings:torch.Tensor, train_class_means:torch.Tensor):
    '''
    Compute the outlier score for the given batch of embeddings and class means obtained from the training set.
    The outlier score for a single datapoint is defined as min_j(||z - m_j||^2), where j is a category and m_j is the mean embedding of this class.
    Parameters
    ----------
    embeddings: a torch.Tensor of shape (N, D) where N is the number of data points and D is the embedding dimension.
    train_class_means: a torch.Tensor of shape (K, D) where K is the number of classes.
    Returns
    -------
    a torch.Tensor of shape (N), representing the outlier score for each of the data points.
    '''
    assert len(embeddings.shape) == 2, f"Expected 2D tensor of shape N ⨉ D (N=datapoints, D=embedding dimension), got {embeddings.shape}"
    assert len(train_class_means.shape) == 2, f"Expected 2D tensor of shape K ⨉ D (K=num_classes, D=embedding dimension), got {train_class_means.shape}"
    # create an expanded version of the embeddings of dimension N ⨉ K ⨉ D, useful for subtracting means
    embeddings_repeated = embeddings.unsqueeze(1).repeat((1, train_class_means.shape[0], 1))
    # compute the difference between the embeddings and the class means
    difference_from_mean = embeddings_repeated - train_class_means
    # compute the squared norm of the difference (N ⨉ K matrix)
    norm_from_mean = difference_from_mean.norm(dim=2)**2
    # get the min for each datapoint
    return norm_from_mean.min(dim=1).values


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--root_train", type=str, default="ImageSet/train")
    parser.add_argument("--root_test", type=str, default="ImageSet/test")
    parser.add_argument("--loss_type", type=str, default="iiloss")
    parser.add_argument("--optimizer", type=str, default="radam")
    parser.add_argument("--out_net", type=int, default=18)
    parser.add_argument("--is_feature_extraction", type=bool, default=True)
    parser.add_argument("--weights_save_path", type=str, default="models/model_BEST2.pt")
    parser.add_argument("--pickle_save_path", type=str, default="out_ii")
    parser.add_argument("--is_ml", type=bool, default=True)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--dim_latent", type=int, default=32)
    parser.add_argument("--lr", type=float, default=.001)
    parser.add_argument("--epochs_lr_decay", nargs="*", type=int, default=[10, 15])
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
        optimizer=args.optimizer,
        dim_latent=args.dim_latent,
        lr=args.lr
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
    extraloader, extraset = data.get_single_dataloader("ImageSet/extra",
                                                        transform_test,
                                                        128,
                                                        balance=False
                                                        ) 

    #define the number of different classes
    num_classes = len(trainset.classes)
    loss_fn = torch.nn.CrossEntropyLoss()
    net = createNet.resNet50Costum(num_classes)
    dict_custom_resnet50, classic = createNet.create_dict_resNet50Costum(net, "resnet50_aug_per_giovanni.pt_resnet50.pt")
    net.load_state_dict(dict_custom_resnet50)
    for param in net.parameters():
        param.requires_grad = True
    if allParams.optimizer.lower() == "sgd":
        optimizer = torch.optim.SGD(net.parameters(),
                                    lr=allParams.lr,
                                    momentum=.9,
                                    weight_decay=5e-4
                                    )
    elif allParams.optimizer.lower() == "radam":
        optimizer = torch.optim.RAdam(net.parameters(), lr=allParams.lr)
    else:
        raise NotImplementedError(f"Invalid optimizer {allParams.optimizer}. Please choose from 'sgd' or 'radam'.")

    # set scheduler
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                     milestones=args.epochs_lr_decay,
                                                     gamma=0.1
                                                     )

    # train
    print('Start Train')
    _, _, = train_model(net,
                      trainloader,
                      loss_fn,
                      optimizer,
                      allParams.get_num_epochs(),
                      lr_scheduler=scheduler,
                      device=allParams.get_device(),
                      loss_type=allParams.get_loss_type(),
                      num_classes=num_classes
                      )
    print('Compute threshold')
    threshold, mean = compute_threshold(net, trainloader, num_classes, allParams.get_device())

    print('Start Test ii loss')
    test_model_iiloss(net,
                        testloader,
                        loss_fn=loss_fn,
                        device=allParams.get_device(),
                        threshold=threshold,
                        mean = mean
                        )
    print('Strat not punches')
    test_model_on_extra(net,
                        extraloader,
                        device=allParams.get_device(),
                        threshold=threshold,
                        mean = mean
                        )

#    outlier_scores_test = eval_outlier_scores(testloader, net, mean, device=allParams.get_device())
#    print(outlier_scores_test)

#    print("Getting outlier scores for ood set")
#    outlier_scores_extra = eval_outlier_scores(extraloader, net, mean, device=allParams.get_device())
#    print(outlier_scores_extra)

    print('Saving weights...')
    os.makedirs(os.path.dirname(allParams.get_weights_save_path()),
                exist_ok=True
                )
    torch.save(net.state_dict(), allParams.get_weights_save_path())
    
    print('Saving pickle')
    utils.save_obj(file_name=f"./pickle_thres_mean_BEST2",
                        first=threshold,
                        second=mean
                        )

print('Finish')
