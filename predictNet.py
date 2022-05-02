import torch
from utils import save_obj


def predictNet(model, testloader):
    feat = []
    feat_labels = []
    model.eval()
    with torch.no_grad(): #TODO mettere to_device?
        for X,y in testloader:
            pred = model(X)
            feat.append(pred)
            feat_labels.append(y)

    save_obj(file_name="normal_prediction_net",
                first=model,
                second=feat,
                third=feat_labels
            )
    return feat, feat_labels