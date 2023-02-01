import torch
import torch.nn as nn
import torch.nn.functional as F
from .features import init_backbone
from .protopnet.model import PPNet
import torch.utils.model_zoo as model_zoo
from collections import OrderedDict

import sys

sys.path.append('lib/protopnet')
print(sys.path, "SYSTEM PATH")

# from model import *

_prototypical_model = {

    "protopnet": PPNet
}


def init_proto_model(manager, classes, backbone, declare_num_features=None):
    """
        Create network with pretrained features and 1x1 convolutional layer

    """
    # Creating tree (backbone+add-on+prototree) architecture

    prototypical_model = backbone.prototypicalModel
    use_chkpt_opt = manager.settingsConfig.useCheckpointOptimizer

    features, trainable_param_names = init_backbone(backbone)
    model = _prototypical_model[prototypical_model](
        num_classes=len(classes), feature_net=features, args=manager.settingsConfig, declare_num_features=declare_num_features)


    # print("PRINTING THE ORIGINAL MODEL LAYERS")
    # for count, param_tensor in enumerate( model.state_dict() ):
    #     print(count, param_tensor, "\t", model.state_dict()[param_tensor].size())
    # print('FINISHED PRINTING ORIGINAL MODEL LAYERS')

    # print()

    # print("PRINTING THE FULLY PRETRAINED MODEL LAYERS")
    # state_dictionary = torch.load("model.pt")
    # for count, param_tensor in enumerate( state_dictionary ):
    #     print(count, param_tensor, "\t", state_dictionary[param_tensor].size())
    # print("FINISHED PRINTING THE FULLY PRETRAINED MODEL LAYERS")

    #----------------------------------------------------------------------------------------------------------------------------------
    # This part is used to change final 4 layer names of pretrained model-> need to correspond to newer version. We needed to save a
    # new model with different keys in the state_dict; All models are already present so no need for this execution; 
    
    # unexpected_keys = ["add_on_layers.0.weight", "add_on_layers.0.bias", "add_on_layers.2.weight", "add_on_layers.2.bias"]
    # correct_keys = ["_add_on.0.weight", "_add_on.0.bias", "_add_on.2.weight", "_add_on.2.bias"]
    # # print(f"LENGTH OF UNEXPECTED KEYS: {len(unexpected_keys)}; LENGTH OF CORRECT KEYS: {len(correct_keys)}")


    # new_model_keys = OrderedDict()
    # assert (len(unexpected_keys) == len(correct_keys)), "amount of unknown keys must be equal to anount of correct keys"

    # for key, value in model.state_dict().items():
    #     for i in range(len(unexpected_keys)):
    #         if key == unexpected_keys[i]:
    #             newKey = correct_keys[i]
    #         else:
    #             newKey = key
    #     new_model_keys[newKey] = value
    
    # torch.save(new_model_keys, "ResNet18_003_mod_layers.pt")
    
    # for key, value in new_model_keys.items():
    #     print(key)

    #----------------------------------------------------------------------------------------------------------------------------------
    # When loading one of the downloaded models (ResNet18_003_mod_layers.pt or ResNet34_003_mod_layers.pt); 
    use_pretrained = False

    # When we use an evaluation script (evaluate_resnet18.job etc)
    use_eval = False
    if use_pretrained:
    # resnet 18/34 is declared in yaml file corresponding to model below: to run just use baselineresnet18 and set corresponding
    # backbone model to either resnet18 or resnet34; additionally, set   numFeatures: 128 for resnet18 and   numFeatures: 256 for resnet34
    # and ofc set use_ptrained to true.

        # model.load_state_dict(torch.load("ResNet34_mod_layers.pt"))
        # proto_vec = model.prototype_vectors


        model.load_state_dict(torch.load("ResNet18_003_mod_layers.pt"))
        
        # proto_vec_2 = model.prototype_vectors
        # print("cheching if proto vectors have been updated")
        # print(proto_vec == proto_vec_2)


        checkpoint = None

    if backbone.loadPath is not None:
        # https://discuss.pytorch.org/t/dataparallel-changes-parameter-names-issue-with-load-state-dict/60211/2
        # https://github.com/pytorch/pytorch/issues/3678
        # https://discuss.pytorch.org/t/how-do-i-update-old-key-name-in-state-dict-to-new-version/76026

        # print("loading the model for KD")

        # checkpoint = checkpoint.load_state_dict(torch.load(backbone.loadPath))
        if not(use_pretrained) and use_eval:
            # this is used for evaluating only and when loading the downloaded pretrained models :
            print(backbone.loadPath)
            model.load_state_dict(torch.load(backbone.loadPath))
            checkpoint = None
        
        # The else statement refers to loading for a kd training process
        else:
            checkpoint = torch.load(backbone.loadPath)
            model = checkpoint['model']
            # print("Loaded model from ", backbone.loadPath)

        if not use_chkpt_opt:
            checkpoint = None
    else:
        # print("checkpoint is None")
        checkpoint = None

    if manager.common.mgpus:
        print("Multi-gpu setting")
        model = nn.DataParallel(model)

    if manager.common.cuda > 0:
        print("Using GPU 222")
        model.cuda()

        print(torch.cuda.is_available())
        device = torch.cuda.current_device()
        print(device)
        print(torch.cuda.get_device_name(device))

    return model, checkpoint, trainable_param_names
