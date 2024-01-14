from . import my_model_utils

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from einops.layers.torch import Rearrange
    
    
class LinearLogitFusionModel(nn.Module):
    def __init__(self, num_classes, pretrained=None):
        super(LinearLogitFusionModel, self).__init__()
        print("Construct LinearLogitFusionModel")

        self.block_A = models.resnet50(weights='DEFAULT')
        self.block_A.fc = nn.Linear(in_features=2048, out_features=2)        
        if pretrained is not None:
            self.block_A.load_state_dict(torch.load(pretrained['model_A']))
            print(f"Load model_A: {pretrained['model_A']} >> Done")
        self.block_A.fc = nn.Identity()

        self.block_B = models.resnet50(weights='DEFAULT')
        self.block_B.fc = nn.Linear(in_features=2048, out_features=2)       
        if pretrained is not None:
            self.block_B.load_state_dict(torch.load(pretrained['model_B']))
            print(f"Load model_B: {pretrained['model_B']} >> Done")
        self.block_B.fc = nn.Identity()

        self.block_C = models.resnet50(weights='DEFAULT')
        self.block_C.fc = nn.Linear(in_features=2048, out_features=2)
        if pretrained is not None:
            self.block_C.load_state_dict(torch.load(pretrained['model_C']))
            print(f"Load model_C: {pretrained['model_C']} >> Done")
        self.block_C.fc = nn.Identity()

        self.fusion = nn.Linear(in_features=2048*3, out_features=2048)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(in_features=2048, out_features=num_classes)
        
    def forward(self, x):
        out_block_A = self.block_A(x)
        out_block_B = self.block_B(x)
        out_block_C = self.block_C(x)
        
        cat = torch.cat((out_block_A, out_block_B, out_block_C), 1)
        y = self.fusion(cat)
        y = self.activation(y)
        y = self.dropout(y)
        y = self.classifier(y)

        return y
    
    
class LinearHeadFusionModel(nn.Module):
    def __init__(self, num_classes, pretrained=None):
        super(LinearHeadFusionModel, self).__init__()
        print("Construct LinearHeadFusionModel")
        
        self.block_A = models.resnet50(weights='DEFAULT')
        self.block_A.fc = nn.Linear(in_features=2048, out_features=2)        
        if pretrained is not None:
            self.block_A.load_state_dict(torch.load(pretrained['model_A']))
            print(f"Load model_A: {pretrained['model_A']} >> Done")

        self.block_B = models.resnet50(weights='DEFAULT')
        self.block_B.fc = nn.Linear(in_features=2048, out_features=2)       
        if pretrained is not None:
            self.block_B.load_state_dict(torch.load(pretrained['model_B']))
            print(f"Load model_B: {pretrained['model_B']} >> Done")

        self.block_C = models.resnet50(weights='DEFAULT')
        self.block_C.fc = nn.Linear(in_features=2048, out_features=2)
        if pretrained is not None:
            self.block_C.load_state_dict(torch.load(pretrained['model_C']))
            print(f"Load model_C: {pretrained['model_C']} >> Done")

        self.classifier = nn.Linear(in_features=2*3, out_features=num_classes)
        
    def forward(self, x):
        out_block_A = self.block_A(x)
        out_block_B = self.block_B(x)
        out_block_C = self.block_C(x)
        
        cat = torch.cat((out_block_A, out_block_B, out_block_C), 1)
        y = self.classifier(cat)

        return y
    
    
class AttentionBasedFusionModel(nn.Module):
    def __init__(self, device, num_classes, pretrained=None):
        super(AttentionBasedFusionModel, self).__init__()
        print("Construct AttentionBasedFusionModel")
        
        self.block_A = models.resnet50(weights='DEFAULT')
        self.block_A.fc = nn.Linear(in_features=2048, out_features=2)        
        if pretrained is not None:
            self.block_A.load_state_dict(torch.load(pretrained['model_A']))
            print(f"Load model_A: {pretrained['model_A']} >> Done")
        self.block_A.fc = nn.Identity()

        self.block_B = models.resnet50(weights='DEFAULT')
        self.block_B.fc = nn.Linear(in_features=2048, out_features=2)       
        if pretrained is not None:
            self.block_B.load_state_dict(torch.load(pretrained['model_B']))
            print(f"Load model_B: {pretrained['model_B']} >> Done")
        self.block_B.fc = nn.Identity()

        self.block_C = models.resnet50(weights='DEFAULT')
        self.block_C.fc = nn.Linear(in_features=2048, out_features=2)
        if pretrained is not None:
            self.block_C.load_state_dict(torch.load(pretrained['model_C']))
            print(f"Load model_C: {pretrained['model_C']} >> Done")
        self.block_C.fc = nn.Identity()
        
        self.fusion = nn.Linear(in_features=2048*3, out_features=2048)
        self.single_head_attention_A = SingleHeadAttention(device, dim=2048, dropout=0.1)
        self.single_head_attention_B = SingleHeadAttention(device, dim=2048, dropout=0.1)
        self.single_head_attention_C = SingleHeadAttention(device, dim=2048, dropout=0.1)
        self.classifier_A = nn.Linear(in_features=2048, out_features=num_classes)
        self.classifier_B = nn.Linear(in_features=2048, out_features=num_classes)
        self.classifier_C = nn.Linear(in_features=2048, out_features=num_classes)
        
    def forward(self, x):
        out_block_A = self.block_A(x)
        out_block_B = self.block_B(x)
        out_block_C = self.block_C(x)
        
        cat = torch.cat((out_block_A, out_block_B, out_block_C), 1)
        fusion = self.fusion(cat)
        
        y_a = self.single_head_attention_A(out_block_A, fusion, fusion)
        y_b = self.single_head_attention_B(out_block_B, fusion, fusion)
        y_c = self.single_head_attention_C(out_block_C, fusion, fusion)
        
        y_a = self.classifier_A(y_a)
        y_b = self.classifier_B(y_b)
        y_c = self.classifier_C(y_c)
        return [y_a, y_b, y_c]
    
    
class TripletLoss(nn.Module):
    def __init__(self, device, num_classes):
        super(TripletLoss, self).__init__()
        self.device = device
        self.num_classes = num_classes
        self.criterion = nn.CrossEntropyLoss()
        
    def forward(self, outputs, targets):
        tar_a = F.one_hot(targets[0], num_classes=self.num_classes).float().to(self.device)
        tar_b = F.one_hot(targets[1], num_classes=self.num_classes).float().to(self.device)
        tar_c = F.one_hot(targets[2], num_classes=self.num_classes).float().to(self.device)
        loss_a = self.criterion(outputs[0], tar_a)
        loss_b = self.criterion(outputs[1], tar_b)
        loss_c = self.criterion(outputs[2], tar_c)
        loss = loss_a + loss_b + loss_c
        return loss, [loss_a, loss_b, loss_c]


class ScaledDotProductAttention(nn.Module):
    def __init__(self, device, dropout=0.0) -> torch.Tensor:
        super().__init__()
        self.dropout = dropout
        self.device  = device
        
    def forward(self, query, key, value):
        L, S = query.size(-2), key.size(-2)
        scale_factor = 1 / math.sqrt(query.size(-1))
        attn_bias = torch.zeros(L, S, dtype=query.dtype).to(self.device)
        attn_weight = query @ key.transpose(-2, -1) * scale_factor
        attn_weight += attn_bias
        attn_weight = torch.softmax(attn_weight, dim=-1)
        attn_weight = torch.dropout(attn_weight, self.dropout, train=True)
        return attn_weight @ value

class SingleHeadAttention(nn.Module):
    def __init__(self, device, dim, dropout) -> torch.Tensor:
        super().__init__()
        self.linear_q = nn.Linear(in_features=dim, out_features=dim)
        self.linear_k = nn.Linear(in_features=dim, out_features=dim)
        self.linear_v = nn.Linear(in_features=dim, out_features=dim)
        self.scaled_dot_product_attention = ScaledDotProductAttention(device, dropout=dropout)
        
    def forward(self, query, key, value):
        q = self.linear_q(query)
        k = self.linear_k(key)
        v = self.linear_v(value)
        return self.scaled_dot_product_attention(q, k, v) 