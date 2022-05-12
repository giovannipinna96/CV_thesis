import pstats
from sklearn.svm import SVC, NuSVC

from utils import scale_features
from sklearn.model_selection import GridSearchCV


class svm_methods():
    """Object used to manage the various types of svm.
    It saves the various svm models fitted on the input data when calling the various functions.
    When this object is created all values ​​are set to None except
    a dictionary containing some default parameters to use the internal function gridSearch()
    """
    def __init__(self):
        self.linear_svm = None
        self.not_linear_svm = None
        self.grid_serach_svm = None
        self.grid_best_parameters = None
        self.param_grid = {
            'C': [0.1, 1, 10, 100, 1000],
            'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
            'kernel': ['rbf']
        }

    def create_linear_svm(self, X, y):
        self.linear_svm = None
        X_std, _ = scale_features(X)
        clf = SVC(gamma='auto')
        clf.fit(X_std, y)
        self.linear_svm = clf

    def predict_linear_svm(self, X):
        return self.linear_svm.predict(X)

    def create_not_linear_svm(self, X, y):
        self.not_linear_svm = None
        X_std, _ = scale_features(X)
        clf = NuSVC(gamma='auto', nu=0.01)
        clf.fit(X_std, y)
        self.not_linear_svm = clf

    def predict_not_linear_svm(self, X):
        return self.not_linear_svm.predict(X)

    def create_grid_serach_svm(self, clf, X, y):
        """Function that calls sklearn's grid Search object.
        The function saves the gridSearch and best parameters inside the object (itselfs)

        Args:
            clf (_type_): classifier to be used in grid Search
            X (_type_): data
            y (_type_): labels
        """
        self.grid_serach_svm = None
        self.grid_best_parameters = None
        X_std, _ = scale_features(X)
        grid_clf = GridSearchCV(clf, self.param_grid)
        grid_clf.fit(X_std, y)
        self.grid_serach_svm = grid_clf
        self.grid_best_parameters = grid_clf.best_params_

    def predict_grid_search_svm(self, X):
        return self.grid_serach_svm.predict(X)

    def get_best_parameters(self):
        return self.get_best_parameters

    def get_param_grid(self):
        return self.param_grid

    def get_linear_svm(self):
        return self.linear_svm

    def get_not_linear_svm(self):
        return self.not_linear_svm

    def get_grid_search_svm(self):
        return self.grid_serach_svm

    def set_param_grid(self, new_param_grid : dict):
        self.get_param_grid = new_param_grid

    def add_value_param_grid(self, key: str, values: list):
        if key in self.param_grid.keys():
            self.param_grid[key] = values
        else:
            print(
                f"the key: {key} not fount in the dictionary, nothing has been changed")
