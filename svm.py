# TODO creare obj come per clustering e fare svm linear and not linear con grid search
import pstats
from sklearn.svm import SVC

from utils import scale_features
from sklearn.model_selection import GridSearchCV


class svm_methods():
    def __init__(self):
        self.linear_svm = None
        self.not_linear_svm = None
        self.grid_serach_svm = None
        self.param_grid = { 
                            'C': [0.1, 1, 10, 100, 1000],
                            'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
                            'kernel': ['rbf']
                            }

    def linear_svm(self):
        pass

    def not_linear_svm(self):
        pass

    def grid_serach_svm(self):
        pass

    def get_param_grid(self):
        return self.param_grid

    def get_linear_svm(self):
        return self.linear_svm

    def get_not_linear_svm(self):
        return self.not_linear_svm

    def get_grid_search_svm(self):
        return self.grid_serach_svm

    def set_param_grid(self, new_param_grid):
        self.get_param_grid = new_param_grid

    def add_value_param_grid(self, key, values : list):
        pass

