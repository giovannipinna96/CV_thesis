import utils

class allParameters():
    def __init__(self, root_train="ImageSet/train",
     root_test="ImageSet/test", weights_save_path="models/model.pt", batch_size_train=32,
      batch_size_test=128, device=utils.use_gpu_if_possible(), model='resnet50',
       pretrained=True, num_epochs=15
       ):
       self.root_train = root_train
       self.root_test = root_test
       self.weights_save_path = weights_save_path
       self.batch_size_train = batch_size_train
       self.batch_size_test = batch_size_test
       self.device = device
       self.model = model
       self.pretrained = pretrained
       self.num_epochs = num_epochs

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


        
