import utils


class allParameters():
    """
    Class that contain all parameters usefull.
    This class has getter and setter for each parameters.
    """

    def __init__(self, root_train="ImageSet/train",
                 root_test="ImageSet/test", weights_save_path="models/model.pt", batch_size_train=32,
                 batch_size_test=128, model='resnet50',
                 pretrained=True, num_epochs=15, not_freeze='nothing', loss_type='crossEntropy'
                 ):
        self.root_train = root_train
        self.root_test = root_test
        self.weights_save_path = weights_save_path
        self.batch_size_train = batch_size_train
        self.batch_size_test = batch_size_test
        self.device = utils.use_gpu_if_possible()
        self.model = model
        self.pretrained = pretrained
        self.num_epochs = num_epochs
        self.not_freeze = not_freeze
        self.loss_type = loss_type

    def get_root_train(self):
        return self.root_train

    def set_root_train(self, root_train):
        self.root_train = root_train

    def get_root_test(self):
        return self.root_test

    def set_root_test(self, root_test):
        self.root_test = root_test

    def get_weights_save_path(self):
        return self.weights_save_path

    def set_weights_save_path(self, weights_save_path):
        self.weights_save_path = weights_save_path

    def get_batch_size_train(self):
        return self.batch_size_train

    def set_batch_size_train(self, batch_size_train):
        self.batch_size_train = batch_size_train

    def get_batch_size_test(self):
        return self.batch_size_test

    def set_batch_size_test(self, batch_size_test):
        self.batch_size_test = batch_size_test

    def get_model(self):
        return self.model

    def set_model(self, model):
        self.model = model

    def get_pretrained(self):
        return self.pretrained

    def set_pretrained(self, pretrained):
        self.pretrained = pretrained

    def get_num_epochs(self):
        return self.num_epochs

    def set_num_epochs(self, num_epochs):
        self.num_epochs = num_epochs

    def get_device(self):
        return self.device

    def get_not_freeze(self):
        return self.not_freeze

    def set_not_freeze(self, not_freeze):
        self.not_freeze = not_freeze

    def get_loss_type(self):
        return self.loss_type

    def set_loss_type(self, loss_type):
        self.loss_type = loss_type
