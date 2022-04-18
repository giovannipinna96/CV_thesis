import torch
import allParameters
import numpy as np
import pandas as pd
import os


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


def save(data, name_file):
    os.makedirs(os.path.dirname(f'data/'), exist_ok=True)
    pd.DataFrame(data).to_csv(f'data/{name_file}.csv')
