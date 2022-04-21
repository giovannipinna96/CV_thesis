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


def load_prestParameters(dictionary: dict):
    """Function used for load all the parameters that are present in the input dictionary

    Args:
        dictionary (dict): dictionary that contain all the parameters to load
    """
    allParams = allParameters(dictionary.get("root_train"),
                              dictionary.get("root_test"),
                              dictionary.get("weights_save_path"),
                              dictionary.get("batch_size_train"),
                              dictionary.get("batch_size_test"),
                              dictionary.get("device"),
                              dictionary.get("model"),
                              dictionary.get("pretrained"),
                              dictionary.get("num_epochs"),
                              dictionary.get("not_freeze")
                              )


def save(data : np.array, name_file : str):
    """Function used for save in csv format the data.
    First of all create or check that the ./data folder exists. If it doesn't exist then it creates it.
    Then it transforms the numpy.array into a pandas DataFrame to save it in csv.

    Args:
        data (numpy.array): data to save
        name_file (str): name of the csv file where the values ​​contained in data will be saved.
    """
    os.makedirs(os.path.dirname(f'data/'), exist_ok=True)
    pd.DataFrame(data).to_csv(f'data/{name_file}.csv')


def scale_features(data):
    """Function that scales data using sklearn's StandardScaler function.(mean=0 and std=1)
    StandardScaler removes the mean and scales the data to unit variance.
    However, the outliers have an influence when computing the empirical mean and standard deviation
    which shrink the range of the feature values as shown in the left figure below. 
    (https://stackoverflow.com/questions/51237635/difference-between-standard-scaler-and-minmaxscaler)

    Args:
        data (_type_): data to scale

    Returns:
        _type_: returns the scaled data
        _type_: returns the StandardScaler object fitted to the data passed as input
    """
    sc = StandardScaler()
    return sc.fit_transform(data), sc.fit(data)


def save_dict_to_json(name_file: str, dictionary: dict):
    tf = open(f"{name_file}.json", "w")
    json.dump(dictionary, tf)
    tf.close()


def save_dict_to_csv(name_file: str, dictionary: dict):
    try:
        with open(f'{name_file}.csv', 'w') as f:
            writer = csv.writer(f)
            for k, v in dictionary.items():
                writer.writerow([k, v])
    except IOError:
        print("I/O error")


def save_obj(file_name : str = "pickle" , **kwargs ):
    """This function allows you to save one or more objects and / or variables in a pickle file.
    The elements are put in a list and then saved

    Args:
        file_name (str, optional): Name of the pickle file. Defaults to "pickle".
    """
    pickle_out = open(f"{file_name}", "wb")
    pickle.dump(list(kwargs.values()), pickle_out)
    pickle_out.close()

def load_pickle_obj(file_name : str = "pickle"):
    with open(f"{file_name}", "rb") as pickle_in:
        pickle_obj = pickle.load(pickle_in)
    return pickle_obj
