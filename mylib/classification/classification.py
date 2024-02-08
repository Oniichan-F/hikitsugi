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


def train_distributed(device, model, train_loader, valid_loader, num_classes, criterion, optimizer, scheduler=None, transforms=None):
    losses, accs = {}, {}
    softmax = nn.Softmax(dim=1)
    
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
            yy = softmax(yy)

            with torch.set_grad_enabled(phase == 'train'):
                outputs  = model(x)
                outputs  = softmax(outputs)
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