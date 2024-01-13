import torch.nn as nn


def setTrainingLayers(model, targets, verbose=False):
    '''
    再学習レイヤーをセット
    '''
    training_params = []
    
    if verbose: print("Setting training parameters")
    
    # 全レイヤーリセット
    for name, param in model.named_parameters():
        param.requires_grad = False
    
    # ターゲットレイヤーのみ再学習に設定
    for name, param in model.named_parameters():
        for target in targets:
            if target in name:
                param.requires_grad = True
                training_params.append(param)
                if verbose: print(f"set training layer: {name}")
    
    if verbose: print("Done\n")
    return training_params


def resetWeights(model):
    if isinstance(model, nn.Linear):
        model.reset_parameters()
        # print(f"Reset weights: {model}")