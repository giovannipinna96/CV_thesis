import torch
from sklearn.preprocessing import StandardScaler
import allParameters


def use_gpu_if_possible():
    return "cuda:0" if torch.cuda.is_available() else "cpu"


def load_prestParameters(dict):
    allParams = allParameters(dict.get("root_train"),
                              dict.get("root_test"),
                              dict.get("weights_save_path"),
                              dict.get("batch_size_train"),
                              dict.get("batch_size_test"),
                              dict.get("device"), dict.get("model"),
                              dict.get("pretrained"),
                              dict.get("num_epochs"),
                              dict.get("not_freeze")
                              )


def scale_features(data):
    sc = StandardScaler()
    return sc.fit_transform(data)
