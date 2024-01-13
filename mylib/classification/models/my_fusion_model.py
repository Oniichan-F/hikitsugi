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
        self.device = device
        
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
    def __init__(self, device, dim, dropout=0.0) -> torch.Tensor:
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
    
    
class MultiHeadAttention(nn.Module):
    def __init__(self, head, dim) -> torch.Tensor:
        super().__init__()
        self.head = head
        self.dim_head = dim // head
        
        self.linear_q = nn.Linear(in_features=dim, out_features=dim)
        self.linear_k = nn.Linear(in_features=dim, out_features=dim)
        self.linear_v = nn.Linear(in_features=dim, out_features=dim)
        
        self.split_head = Rearrange("b n (h d) -> b h n d", h = self.head)
        self.softmax = nn.Softmax(dim=-1)
        self.concat = Rearrange("b h n d -> b n (h d)", h = self.head)

    def forward(self, query, key, value):
        q = self.linear_q(query)
        k = self.linear_k(key)
        v = self.linear_v(value)
        
        q = torch.reshape(q, (q.shape[0], 32, 32))
        k = torch.reshape(k, (k.shape[0], 32, 32))
        v = torch.reshape(v, (v.shape[0], 32, 32))

        q = self.split_head(q)
        k = self.split_head(k)
        v = self.split_head(v)

        logit = torch.matmul(q, k.transpose(-1, -2)) * (self.dim_head ** -0.5)
        attn_weight = self.softmax(logit)
        output = torch.matmul(attn_weight, v)
        output = self.concat(output)
        output = torch.reshape(output, (output.shape[0], 32*32))
        return output
       
        
class MLP(nn.Module):
    def __init__(self, dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features=dim, out_features=dim*2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(in_features=dim*2, out_features=dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        return self.net(x)
    
    
class TransformerEncoder_MHA(nn.Module):
    def __init__(self, dim, head):
        super().__init__()
        self.norm_q = nn.LayerNorm(dim)
        self.norm_k = nn.LayerNorm(dim)
        self.norm_v = nn.LayerNorm(dim)
        self.norm_mlp = nn.LayerNorm(dim)
        self.multi_head_attention = MultiHeadAttention(head=head, dim=dim)
        self.mlp = MLP(dim=dim, dropout=0.1)
        
    def forward(self, q, k, v):
        q = self.multi_head_attention(self.norm_q(q), self.norm_k(k), self.norm_v(v)) + q
        q = self.mlp(self.norm_mlp(q)) + q
        return q
    

class TransformerEncoder_SHA(nn.Module):
    def __init__(self, device, dim):
        super().__init__()
        self.norm_q = nn.LayerNorm(dim)
        self.norm_k = nn.LayerNorm(dim)
        self.norm_v = nn.LayerNorm(dim)
        self.norm_mlp = nn.LayerNorm(dim)
        self.single_head_attention = SingleHeadAttention(device=device, dim=dim, dropout=0.2)
        self.mlp = MLP(dim=dim, dropout=0.1)
    
    def forward(self, q, k, v):
        q = self.single_head_attention(self.norm_q(q), self.norm_k(k), self.norm_v(v)) + q
        q = self.mlp(self.norm_mlp(q)) + q

        return q
       
        
class FusionModel_MHA(nn.Module):
    def __init__(self, num_classes, pretrained=None, ckpt_head_size=2, dim=1024, head=4, depth=2, mode='FFF'):
        super(FusionModel_MHA, self).__init__()
        
        self.head = head
        self.depth = depth
        self.mode = mode
        
        self.block_A = models.resnet50(weights='DEFAULT')
        self.block_A.fc = nn.Linear(in_features=2048, out_features=ckpt_head_size)
        if pretrained is not None:
            self.block_A.load_state_dict(torch.load(pretrained['model_A']))
            self.block_A.fc = nn.Linear(in_features=2048, out_features=dim)
            print(f"Load A-model: {pretrained['model_A']} >> Done")
        
        self.block_B = models.resnet50(weights='DEFAULT')
        self.block_B.fc = nn.Linear(in_features=2048, out_features=ckpt_head_size)
        if pretrained is not None:
            self.block_B.load_state_dict(torch.load(pretrained['model_B']))
            self.block_B.fc = nn.Linear(in_features=2048, out_features=dim)
            print(f"Load B-model: {pretrained['model_B']} >> Done")
            
        self.block_C = models.resnet50(weights='DEFAULT')
        self.block_C.fc = nn.Linear(in_features=2048, out_features=ckpt_head_size)
        if pretrained is not None:
            self.block_C.load_state_dict(torch.load(pretrained['model_C']))
            self.block_C.fc = nn.Linear(in_features=2048, out_features=dim)
            print(f"Load C-model: {pretrained['model_C']} >> Done")
            
        self.wisdom_fusion = nn.Linear(in_features=dim*3, out_features=dim)
        self.transformer_encoders = nn.ModuleList([TransformerEncoder_MHA(dim=dim, head=head) for _ in range(depth)])
        self.norm_classifier = nn.LayerNorm(dim)
        self.classifier = nn.Linear(in_features=dim, out_features=num_classes)
                
    def forward(self, x):
        out_block_A = self.block_A(x)
        out_block_B = self.block_B(x)
        out_block_C = self.block_C(x)
        
        fusion = self.wisdom_fusion(torch.cat((out_block_A, out_block_B, out_block_C), 1))

        if self.mode == 'FFF':
            q = fusion
            k = fusion
            v = fusion
            
        elif self.mode == 'AAA':
            q = out_block_A
            k = out_block_A
            v = out_block_A
        elif self.mode == 'AAF':
            q = out_block_A
            k = out_block_A
            v = fusion
        elif self.mode == 'AFA':
            q = out_block_A
            k = fusion
            v = out_block_A
        elif self.mode == 'FAA':
            q = fusion
            k = out_block_A
            v = out_block_A
        elif self.mode == 'FFA':
            q = fusion
            k = fusion
            v = out_block_A
        elif self.mode == 'FAF':
            q = fusion
            k = out_block_A
            v = fusion
        elif self.mode == 'AFF':
            q = out_block_A
            k = fusion
            v = fusion
            
        elif self.mode == 'BBB':
            q = out_block_B
            k = out_block_B
            v = out_block_B
        elif self.mode == 'BBF':
            q = out_block_B
            k = out_block_B
            v = fusion
        elif self.mode == 'BFB':
            q = out_block_B
            k = fusion
            v = out_block_B
        elif self.mode == 'FBB':
            q = fusion
            k = out_block_B
            v = out_block_B
        elif self.mode == 'FFB':
            q = fusion
            k = fusion
            v = out_block_B
        elif self.mode == 'FBF':
            q = fusion
            k = out_block_B
            v = fusion
        elif self.mode == 'BFF':
            q = out_block_B
            k = fusion
            v = fusion

        elif self.mode == 'CCC':
            q = out_block_C
            k = out_block_C
            v = out_block_C
        elif self.mode == 'CCF':
            q = out_block_C
            k = out_block_C
            v = fusion
        elif self.mode == 'CFC':
            q = out_block_C
            k = fusion
            v = out_block_C
        elif self.mode == 'FCC':
            q = fusion
            k = out_block_C
            v = out_block_C
        elif self.mode == 'FFC':
            q = fusion
            k = fusion
            v = out_block_C
        elif self.mode == 'FCF':
            q = fusion
            k = out_block_C
            v = fusion
        elif self.mode == 'CFF':
            q = out_block_C
            k = fusion
            v = fusion
        
        for transformer_encoder in self.transformer_encoders:
            q = transformer_encoder(q, k, v)
        
        y = self.classifier(self.norm_classifier(q))

        return y
    
    
class FusionModel_SHA(nn.Module):
    def __init__(self, num_classes, device, pretrained=None, ckpt_head_size=2, dim=1024, depth=1, mode='FFF'):
        super(FusionModel_SHA, self).__init__()
        
        self.depth = depth
        self.mode = mode
        
        self.block_A = models.resnet50(weights='DEFAULT')
        self.block_A.fc = nn.Linear(in_features=2048, out_features=ckpt_head_size)
        if pretrained is not None:
            self.block_A.load_state_dict(torch.load(pretrained['model_A']))
            self.block_A.fc = nn.Linear(in_features=2048, out_features=dim)
            print(f"Load A-model: {pretrained['model_A']} >> Done")
        
        self.block_B = models.resnet50(weights='DEFAULT')
        self.block_B.fc = nn.Linear(in_features=2048, out_features=ckpt_head_size)
        if pretrained is not None:
            self.block_B.load_state_dict(torch.load(pretrained['model_B']))
            self.block_B.fc = nn.Linear(in_features=2048, out_features=dim)
            print(f"Load B-model: {pretrained['model_B']} >> Done")
            
        self.block_C = models.resnet50(weights='DEFAULT')
        self.block_C.fc = nn.Linear(in_features=2048, out_features=ckpt_head_size)
        if pretrained is not None:
            self.block_C.load_state_dict(torch.load(pretrained['model_C']))
            self.block_C.fc = nn.Linear(in_features=2048, out_features=dim)
            print(f"Load C-model: {pretrained['model_C']} >> Done")
            
        self.wisdom_fusion = nn.Linear(in_features=dim*3, out_features=dim)
        self.transformer_encoders = nn.ModuleList([TransformerEncoder_SHA(device=device, dim=dim) for _ in range(depth)])
        self.norm_classifier = nn.LayerNorm(dim)
        self.classifier = nn.Linear(in_features=dim, out_features=num_classes)
                
    def forward(self, x):
        out_block_A = self.block_A(x)
        out_block_B = self.block_B(x)
        out_block_C = self.block_C(x)
        
        fusion = self.wisdom_fusion(torch.cat((out_block_A, out_block_B, out_block_C), 1))

        if self.mode == 'FFF':
            q = fusion
            k = fusion
            v = fusion
            
        elif self.mode == 'AAA':
            q = out_block_A
            k = out_block_A
            v = out_block_A
        elif self.mode == 'AAF':
            q = out_block_A
            k = out_block_A
            v = fusion
        elif self.mode == 'AFA':
            q = out_block_A
            k = fusion
            v = out_block_A
        elif self.mode == 'FAA':
            q = fusion
            k = out_block_A
            v = out_block_A
        elif self.mode == 'FFA':
            q = fusion
            k = fusion
            v = out_block_A
        elif self.mode == 'FAF':
            q = fusion
            k = out_block_A
            v = fusion
        elif self.mode == 'AFF':
            q = out_block_A
            k = fusion
            v = fusion
            
        elif self.mode == 'BBB':
            q = out_block_B
            k = out_block_B
            v = out_block_B
        elif self.mode == 'BBF':
            q = out_block_B
            k = out_block_B
            v = fusion
        elif self.mode == 'BFB':
            q = out_block_B
            k = fusion
            v = out_block_B
        elif self.mode == 'FBB':
            q = fusion
            k = out_block_B
            v = out_block_B
        elif self.mode == 'FFB':
            q = fusion
            k = fusion
            v = out_block_B
        elif self.mode == 'FBF':
            q = fusion
            k = out_block_B
            v = fusion
        elif self.mode == 'BFF':
            q = out_block_B
            k = fusion
            v = fusion

        elif self.mode == 'CCC':
            q = out_block_C
            k = out_block_C
            v = out_block_C
        elif self.mode == 'CCF':
            q = out_block_C
            k = out_block_C
            v = fusion
        elif self.mode == 'CFC':
            q = out_block_C
            k = fusion
            v = out_block_C
        elif self.mode == 'FCC':
            q = fusion
            k = out_block_C
            v = out_block_C
        elif self.mode == 'FFC':
            q = fusion
            k = fusion
            v = out_block_C
        elif self.mode == 'FCF':
            q = fusion
            k = out_block_C
            v = fusion
        elif self.mode == 'CFF':
            q = out_block_C
            k = fusion
            v = fusion
        
        for transformer_encoder in self.transformer_encoders:
            q = transformer_encoder(q, k, v)
        
        y = self.classifier(self.norm_classifier(q))

        return y