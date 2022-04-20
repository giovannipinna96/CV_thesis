import torch
from sklearn.preprocessing import StandardScaler
import allParameters
import numpy as np
import pandas as pd
import os
import csv
import json
import pickle


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


def scale_features(data):
    sc = StandardScaler()
    return sc.fit_transform(data), sc.fit(data)


def save_dict_json(name_file: str, dictionary: dict):
    tf = open(f"{name_file}.json", "w")
    json.dump(dictionary, tf)
    tf.close()


def save_dict_csv(name_file: str, dictionary: dict):
    try:
        with open(f'{name_file}.csv', 'w') as f:
            writer = csv.writer(f)
            for k, v in dictionary.items():
                writer.writerow([k, v])
    except IOError:
        print("I/O error")


def save_obj(file_name : str = "pickle" , **kwargs ):
    pickle_out = open(f"{file_name}", "wb")
    pickle.dump(list(kwargs.values()), pickle_out)
    pickle_out.close()

def load_pickle_obj(file_name : str = "pickle"):
    with open(f"{file_name}", "rb") as pickle_in:
        pickle_obj = pickle.load(pickle_in)
    return pickle_obj
