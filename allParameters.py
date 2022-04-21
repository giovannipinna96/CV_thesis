import utils


class allParameters():
    """
    Class that contain all parameters usefull.
    This class has getter and setter for each parameters.
    """

    def __init__(self, root_train="ImageSet/train",
                 root_test="ImageSet/test", weights_save_path="models/model.pt", batch_size_train=32,
                 batch_size_test=128, model='resnet50',
                 pretrained=True, num_epochs=15, not_freeze='nothing', loss_type='crossEntropy',
                 out_net=18, is_feature_extraction=True
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
        self.out_net = out_net
        self.is_feature_extraction = is_feature_extraction

    def get_out_net(self):
        return self.out_net

    def set_out_net(self, out_net: int):
        self.out_net = out_net

    def get_is_feature_extraction(self):
        return self.is_feature_extraction

    def set_is_feature_extraction(self, is_feature_extraction: bool):
        """Function used to see if the user wants to execute the code also to extract the features
        from the various intermediate layers of the network.
        This was done because that part of the code can be computationally expensive.

        Args:
            is_feature_extraction (bool): if True it will execute the code to extract the features from the intermediate layers of the network
        """
        self.is_feature_extraction = is_feature_extraction

    def get_root_train(self):
        return self.root_train

    def set_root_train(self, root_train: str):
        self.root_train = root_train

    def get_root_test(self):
        return self.root_test

    def set_root_test(self, root_test: str):
        self.root_test = root_test

    def get_weights_save_path(self):
        return self.weights_save_path

    def set_weights_save_path(self, weights_save_path: str):
        self.weights_save_path = weights_save_path

    def get_batch_size_train(self):
        return self.batch_size_train

    def set_batch_size_train(self, batch_size_train: int):
        self.batch_size_train = batch_size_train

    def get_batch_size_test(self):
        return self.batch_size_test

    def set_batch_size_test(self, batch_size_test: int):
        self.batch_size_test = batch_size_test

    def get_model(self):
        return self.model

    def set_model(self, model):
        self.model = model

    def get_pretrained(self):
        return self.pretrained

    def set_pretrained(self, pretrained: bool):
        """Function to set the pretrained value.
        This value allows us to decide if we want to use a network trained with ImageNet weights
         or if we want to start an entire workout from scratch.

        Args:
            pretrained (bool): if True will use the net for ImageNet weight training
        """
        self.pretrained = pretrained

    def get_num_epochs(self):
        return self.num_epochs

    def set_num_epochs(self, num_epochs: int):
        self.num_epochs = num_epochs

    def get_device(self):
        return self.device

    def get_not_freeze(self):
        return self.not_freeze

    def set_not_freeze(self, not_freeze: list):
        """Function to set the not_freeze value in the object of type allParameters ().
        In the not_freeze list we insert all the layers of the network that we will have required_grad = True

        Args:
            not_freeze (list): list where insert all the layers of the network that we will have required_grad = True
        """
        self.not_freeze = not_freeze

    def get_loss_type(self):
        return self.loss_type

    def set_loss_type(self, loss_type: str):
        self.loss_type = loss_type
