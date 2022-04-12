#preset parameters
import utils

resNet50 = {
    "root_train" : "ImageSet/train",
    "root_test" : "ImageSet/test",
    "weights_save_path" : "models/model.pt",
    "batch_size_train" : 32,
    "batch_size_test" : 128,
    "model" : 'resnet50',
    "pretrained" : True,
    "num_epochs" : 15,
    "not_freeze" : 'nothing'
    }

resNet18 = {
    "root_train" : "ImageSet/train",
    "root_test" : "ImageSet/test",
    "weights_save_path" : "models/model.pt",
    "batch_size_train" : 32,
    "batch_size_test" : 128,
    "model" : 'resnet18',
    "pretrained" : True,
    "num_epochs" : 15,
    "not_freeze" : 'nothing'
    }

resNet101 = {
    "root_train" : "ImageSet/train",
    "root_test" : "ImageSet/test",
    "weights_save_path" : "models/model.pt",
    "batch_size_train" : 32,
    "batch_size_test" : 128,
    "model" : 'resnet101',
    "pretrained" : True,
    "num_epochs" : 15,
    "not_freeze" : 'nothing'
    }

vgg16 = {
    "root_train" : "ImageSet/train",
    "root_test" : "ImageSet/test",
    "weights_save_path" : "models/model.pt",
    "batch_size_train" : 32,
    "batch_size_test" : 128,
    "model" : 'vgg16',
    "pretrained" : True,
    "num_epochs" : 15,
    "not_freeze" : 'nothing'
    }

