import torch
import torch.nn as nn
import torch.nn.functional as F


def train(device, model, train_loader, valid_loader, num_classes, criterion, optimizer, scheduler=None, transforms=None):
    losses, accs = {}, {}
    
    for phase in ['train', 'valid']:
        ep_loss = 0.0
        correct = 0
        
        if phase == 'train':
            model.train()
            loader = train_loader
        elif phase == 'valid':
            model.eval()
            loader = valid_loader
            
        for x, y, _ in loader:
            optimizer.zero_grad()
            
            if transforms is not None:
                x = transforms(x)
            
            x = x.float().to(device)
            y = F.one_hot(y, num_classes=num_classes).float().to(device)
            
            with torch.set_grad_enabled(phase == 'train'):
                outputs  = model(x)
                loss     = criterion(outputs, y)
                _, preds = torch.max(outputs, 1)
                
                if phase == 'train':
                    loss.backward()
                    optimizer.step()
                    
                ep_loss += loss.item() * x.size(0)
                correct += torch.sum(preds == torch.argmax(y, dim=1))
                
        losses[phase] = ep_loss / len(loader.dataset)
        accs[phase]   = (correct.double() / len(loader.dataset)).to('cpu').detach().clone().numpy()[()]
    
    if scheduler is not None: scheduler.step()
    
    return losses, accs


def train2(device, model, train_loader, valid_loader, num_classes, criterion, optimizer, scheduler=None, transforms=None):
    losses, accs = {}, {}
    
    for phase in ['train', 'valid']:
        ep_loss = 0.0
        correct = 0
        
        if phase == 'train':
            model.train()
            loader = train_loader
        elif phase == 'valid':
            model.eval()
            loader = valid_loader
            
        for x, y, _ in loader:
            optimizer.zero_grad()
            
            if transforms is not None:
                x = transforms(x)
            
            x = x.float().to(device)
            
            yy = []
            y0 = y[0].tolist()
            y1 = y[1].tolist()
            for i in range(len(y0)):
                yy.append([y0[i], y1[i]])
            yy = torch.tensor(yy).float().to(device)
            
            with torch.set_grad_enabled(phase == 'train'):
                outputs  = model(x)
                loss     = criterion(outputs, yy)
                _, preds = torch.max(outputs, 1)
                
                if phase == 'train':
                    loss.backward()
                    optimizer.step()
                    
                ep_loss += loss.item() * x.size(0)
                correct += torch.sum(preds == torch.argmax(yy, dim=1))
                
        losses[phase] = ep_loss / len(loader.dataset)
        accs[phase]   = (correct.double() / len(loader.dataset)).to('cpu').detach().clone().numpy()[()]
    
    if scheduler is not None: scheduler.step()
    
    return losses, accs


def train3(device, model, train_loader, valid_loader, num_classes, criterion, optimizer, scheduler=None, transforms=None):
    losses = {}
    losses_A, losses_B, losses_C = {}, {}, {}
    
    for phase in ['train', 'valid']:
        ep_loss, ep_loss_A, ep_loss_B, ep_loss_C = 0.0, 0.0, 0.0, 0.0
        correct = 0
        
        if phase == 'train':
            model.train()
            loader = train_loader
        elif phase == 'valid':
            model.eval()
            loader = valid_loader
            
        for x, y, _ in loader:
            optimizer.zero_grad()
            
            if transforms is not None:
                x = transforms(x)
            
            x = x.float().to(device)
            
            with torch.set_grad_enabled(phase == 'train'):
                outputs = model(x)
                loss, loss_sub  = criterion(outputs, y)
                
                if phase == 'train':
                    loss.backward()
                    optimizer.step()
                    
                ep_loss += loss.item() * x.size(0)
                ep_loss_A += loss_sub[0] * x.size(0)
                ep_loss_B += loss_sub[1] * x.size(0)
                ep_loss_C += loss_sub[2] * x.size(0)
                
        losses[phase] = ep_loss / len(loader.dataset)
        losses_A[phase] = ep_loss_A / len(loader.dataset)
        losses_B[phase] = ep_loss_B / len(loader.dataset)
        losses_C[phase] = ep_loss_C / len(loader.dataset)
    
    if scheduler is not None: scheduler.step()
    
    return losses, [losses_A, losses_B, losses_C]


def test(device, model, test_loader, name_classes):
    fnames, trues, preds, probs = [], [], [], []
    softmax = nn.Softmax(dim=1)
    
    model.eval()
    
    for x, y, f in test_loader:
        x = x.float().to(device)
        y = y.float().to(device)
        
        with torch.set_grad_enabled(False):
            outputs = model(x)
            prob    = softmax(outputs)
            _, pred = torch.max(outputs, 1)
        
        fnames += [l for l in f]
        trues  += [int(l) for l in y]
        preds  += [int(l) for l in pred]    
        probs  += [l.to('cpu').detach().clone().numpy().tolist() for l in prob]
        
    results = {}
    results['fname'] = fnames
    results['true']  = trues
    results['pred']  = preds
    for name in name_classes:
        results[name] = []
    for i in range(len(probs)):
        j = 0
        for name in name_classes:
            results[name].append(probs[i][j])
            j += 1
     
    return results