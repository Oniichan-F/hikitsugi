from . import my_model_utils

import torch.nn as nn
from torchvision import models


# 再学習レイヤー
VGG16_CLASSIFIER      = ['classifier.0.weight', 'classifier.0.bias', 'classifier.2.weight', 'classifier.2.bias']
RESNET_CLASSIFIER     = ['fc']
RESNET_LAYERS         = ['conv1', 'bn1', 'relu', 'maxpool', 'layer1', 'layer2', 'layer3' 'layer4', 'fc']
RESNEXT_CLASSIFIER    = ['fc']
RESNET_LAYERS         = ['conv1', 'bn1', 'relu', 'maxpool', 'layer1', 'layer2', 'layer3' 'layer4', 'fc']
EFFICIENTNET_CLASSIFIER = ['classifier']
EFFICIENTNET_LAYERS     = ['features.0', 'features.1', "~", 'features.8', 'classifier']
CONVNEXT_BASE_CLASSIFIER = ['classifier']
CONVNEXT_BASE_LAYERS     = ['features.0', '~', 'features.7', 'classifier']
'''
B0 : 224
B1 : 240
B2 : 260
B3 : 300
B4 : 380
B5 : 456
B6 : 528
B7 : 600
'''


def build_VGG16(num_classes, training_layers=VGG16_CLASSIFIER, pretrained=True, features=1024, verbose=False):
    model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
    model.classifier = nn.Sequential(*list(model.classifier.children())[:-4])
    model.classifier[0] = nn.Linear(in_features=25088, out_features=features)
    model.classifier[2] = nn.Linear(in_features=features, out_features=num_classes)  
    params = my_model_utils.setTrainingLayers(model, training_layers, verbose)
    
    return model, params

def build_ResNet50(num_classes, training_layers=RESNET_CLASSIFIER, pretrained=True, verbose=False):
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    model.fc = nn.Linear(in_features=2048, out_features=num_classes)   
    params = my_model_utils.setTrainingLayers(model, training_layers, verbose)
    
    return model, params

def build_ResNet101(num_classes, training_layers=RESNET_CLASSIFIER, pretrained=True, verbose=False):
    model = models.resnet101(weights=models.ResNet101_Weights.IMAGENET1K_V2)
    model.fc = nn.Linear(in_features=2048, out_features=num_classes)   
    params = my_model_utils.setTrainingLayers(model, training_layers, verbose)
    
    return model, params

def build_ResNet152(num_classes, training_layers=RESNET_CLASSIFIER, pretrained=True, verbose=False):
    model = models.resnet152(weights=models.ResNet152_Weights.IMAGENET1K_V2)
    model.fc = nn.Linear(in_features=2048, out_features=num_classes)   
    params = my_model_utils.setTrainingLayers(model, training_layers, verbose)
    
    return model, params

def build_ResNeXt50(num_classes, training_layers=RESNEXT_CLASSIFIER, pretrained=True, verbose=False):
    model = models.resnext50_32x4d(weights=models.ResNeXt50_32X4D_Weights.IMAGENET1K_V2)
    model.fc = nn.Linear(in_features=2048, out_features=num_classes)   
    params = my_model_utils.setTrainingLayers(model, training_layers, verbose)
    
    return model, params

def build_ResNeXt101(num_classes, training_layers=RESNEXT_CLASSIFIER, pretrained=True, verbose=False):
    model = models.resnext101_32x8d(weights=models.ResNeXt101_32X8D_Weights.IMAGENET1K_V2)
    model.fc = nn.Linear(in_features=2048, out_features=num_classes)   
    params = my_model_utils.setTrainingLayers(model, training_layers, verbose)
    
    return model, params


def build_EfficientNet_B0(num_classes, training_layers=EFFICIENTNET_CLASSIFIER, pretrained=True, verbose=False):
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
    model.classifier[1] = nn.Linear(in_features=1280, out_features=num_classes)
    params = my_model_utils.setTrainingLayers(model, training_layers, verbose)
    
    return model, params

def build_EfficientNet_B1(num_classes, training_layers=EFFICIENTNET_CLASSIFIER, pretrained=True, verbose=False):
    model = models.efficientnet_b1(weights=models.EfficientNet_B1_Weights.IMAGENET1K_V1)
    model.classifier[1] = nn.Linear(in_features=1280, out_features=num_classes)
    params = my_model_utils.setTrainingLayers(model, training_layers, verbose)
    
    return model, params

def build_EfficientNet_B2(num_classes, training_layers=EFFICIENTNET_CLASSIFIER, pretrained=True, verbose=False):
    model = models.efficientnet_b2(weights=models.EfficientNet_B2_Weights.IMAGENET1K_V1)
    model.classifier[1] = nn.Linear(in_features=1408, out_features=num_classes)
    params = my_model_utils.setTrainingLayers(model, training_layers, verbose)
    
    return model, params

def build_EfficientNet_B3(num_classes, training_layers=EFFICIENTNET_CLASSIFIER, pretrained=True, verbose=False):
    model = models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.IMAGENET1K_V1)
    model.classifier[1] = nn.Linear(in_features=1536, out_features=num_classes)
    params = my_model_utils.setTrainingLayers(model, training_layers, verbose)
    
    return model, params

def build_EfficientNet_B4(num_classes, training_layers=EFFICIENTNET_CLASSIFIER, pretrained=True, verbose=False):
    model = models.efficientnet_b4(weights=models.EfficientNet_B4_Weights.IMAGENET1K_V1)
    model.classifier[1] = nn.Linear(in_features=1792, out_features=num_classes)
    params = my_model_utils.setTrainingLayers(model, training_layers, verbose)
    
    return model, params

def build_EfficientNet_B5(num_classes, training_layers=EFFICIENTNET_CLASSIFIER, pretrained=True, verbose=False):
    model = models.efficientnet_b5(weights=models.EfficientNet_B5_Weights.IMAGENET1K_V1)
    model.classifier[1] = nn.Linear(in_features=2048, out_features=num_classes)
    params = my_model_utils.setTrainingLayers(model, training_layers, verbose)
    
    return model, params

def build_EfficientNet_B6(num_classes, training_layers=EFFICIENTNET_CLASSIFIER, pretrained=True, verbose=False):
    model = models.efficientnet_b6(weights=models.EfficientNet_B6_Weights.IMAGENET1K_V1)
    model.classifier[1] = nn.Linear(in_features=2304, out_features=num_classes)
    params = my_model_utils.setTrainingLayers(model, training_layers, verbose)
    
    return model, params

def build_EfficientNet_B7(num_classes, training_layers=EFFICIENTNET_CLASSIFIER, pretrained=True, verbose=False):
    model = models.efficientnet_b7(weights=models.EfficientNet_B7_Weights.IMAGENET1K_V1)
    model.classifier[1] = nn.Linear(in_features=2560, out_features=num_classes)
    params = my_model_utils.setTrainingLayers(model, training_layers, verbose)
    
    return model, params

def build_ConvNeXt_Tiny(num_classes, training_layers=CONVNEXT_BASE_CLASSIFIER, pretrained=True, verbose=False):
    model = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
    model.classifier[2] = nn.Linear(in_features=768, out_features=num_classes)   
    params = my_model_utils.setTrainingLayers(model, training_layers, verbose)
    
    return model, params

def build_ConvNeXt_Small(num_classes, training_layers=CONVNEXT_BASE_CLASSIFIER, pretrained=True, verbose=False):
    model = models.convnext_small(weights=models.ConvNeXt_Small_Weights.IMAGENET1K_V1)
    model.classifier[2] = nn.Linear(in_features=768, out_features=num_classes)   
    params = my_model_utils.setTrainingLayers(model, training_layers, verbose)
    
    return model, params

def build_ConvNeXt_Base(num_classes, training_layers=CONVNEXT_BASE_CLASSIFIER, pretrained=True, verbose=False):
    model = models.convnext_base(weights=models.ConvNeXt_Base_Weights.IMAGENET1K_V1)
    model.classifier[2] = nn.Linear(in_features=1024, out_features=num_classes)   
    params = my_model_utils.setTrainingLayers(model, training_layers, verbose)
    
    return model, params