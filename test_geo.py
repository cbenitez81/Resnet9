# Supporting functions for training and testing
import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from torchvision.datasets import ImageFolder
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
from model_rn import  ResNet9
from PIL import Image
import argparse
import os


def get_default_device():
    """Pick GPU if available, else CPU"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')
    
def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list,tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)

class DeviceDataLoader():
    """Wrap a dataloader to move data to a device"""
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device
        
    def __iter__(self):
        """Yield a batch of data after moving it to device"""
        for b in self.dl: 
            yield to_device(b, self.device)

    def __len__(self):
        """Number of batches"""
        return len(self.dl)

@torch.no_grad()
def evaluate(model, val_loader):
    model.eval()
    outputs = [model.validation_step(batch) for batch in val_loader]
    return model.validation_epoch_end(outputs)

@torch.no_grad()
def get_embedding(model, val_loader):
    model.eval()
    outputs = [model.get_embedding(batch) for batch in val_loader]
    return outputs

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def fit_one_cycle(epochs, max_lr, model, train_loader, val_loader, 
                  weight_decay=0, grad_clip=None, opt_func=torch.optim.SGD):
    torch.cuda.empty_cache()
    history = []
    
    # Set up cutom optimizer with weight decay
    optimizer = opt_func(model.parameters(), max_lr, weight_decay=weight_decay)
    # Set up one-cycle learning rate scheduler
    sched = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr, epochs=epochs, 
                                                steps_per_epoch=len(train_loader))
    
    for epoch in range(epochs):
        # Training Phase 
        model.train()
        train_losses = []
        lrs = []
        k1 = 0
        for batch in train_loader:
            loss = model.training_step(batch)
            train_losses.append(loss)
            loss.backward()
            if k1 % 100 == 0:
                print(f'batch {k1}')  
            k1 += 1
            # Gradient clipping
            if grad_clip: 
                nn.utils.clip_grad_value_(model.parameters(), grad_clip)
            
            optimizer.step()
            optimizer.zero_grad()
            
            # Record & update learning rate
            lrs.append(get_lr(optimizer))
            sched.step()
        
        # Validation phase
        result = evaluate(model, val_loader)
        result['train_loss'] = torch.stack(train_losses).mean().item()
        result['lrs'] = lrs
        model.epoch_end(epoch, result)
        history.append(result)
    return history

def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


def train_network(model, data_dir, batch_size = 100, epochs = 15, device = torch.device('cuda')):
    transform_train = transforms.Compose(
        [transforms.ToTensor(), transforms.Resize((32,32)),    
         transforms.RandomHorizontalFlip()])
    
    transform_test = transforms.Compose(
        [transforms.ToTensor(), transforms.Resize((32,32))])

    
    trainset = ImageFolder(data_dir+'/set_train/', transform_train)
    testset = ImageFolder(data_dir+'/set_test/', transform_test)
    trainset_no_aug = ImageFolder(data_dir+'/set_train/', transform_test)
        
    trainloader = DataLoader(trainset, batch_size, shuffle=True, num_workers=3, pin_memory=True)
    testloader = DataLoader(testset, batch_size, num_workers=3, pin_memory=True)
    trainloader = DeviceDataLoader(trainloader, device)
    testloader = DeviceDataLoader(testloader, device)
    
    trainloader_no_aug = DataLoader(trainset_no_aug, batch_size, num_workers=3, pin_memory=True)
    trainloader_no_aug = DeviceDataLoader(trainloader_no_aug, device)
    
    max_lr = 0.01
    grad_clip = 0.1
    weight_decay = 1e-4
    opt_func = torch.optim.Adam
    
    
    history = fit_one_cycle(epochs, max_lr, model, trainloader, testloader, 
                                 grad_clip=grad_clip, 
                                 weight_decay=weight_decay, 
                                 opt_func=opt_func)
    
    torch.save(model.state_dict(), 'model_weights.pth')
    train_embedding = torch.cat(get_embedding(model,trainloader_no_aug))
    torch.save(train_embedding, 'database.pt')
    
    return history
    

#%%

def main():
    
    def show_test_sample_k(testset, test_id = 1000, k = 5):
        train_embedding = torch.load('database.pt')
        _, test_emb = model(testset[test_id][0].unsqueeze(0).cuda())
    
        norms = (train_embedding - test_emb).norm(dim=1).argsort()
        
        grid = [trainset_no_aug[b][0] for b in norms[0:k]]
        grid = [testset[test_id][0]] + grid 
        imshow(make_grid(grid))    
        
    def show_sample_k(img, k = 5):
        #train_embedding = torch.load('database.pt')
        train_embedding = torch.cat(get_embedding(model,trainloader_no_aug))
        x = transform_test(img)
        x.unsqueeze_(0)
        
        _, test_emb = model(x.cuda())
    
        norms = (train_embedding - test_emb).norm(dim=1).argsort()
        
        grid = [trainset_no_aug[b][0] for b in norms[0:k]]
        grid = [x.squeeze_(0)] + grid 
        imshow(make_grid(grid))  

    parser = argparse.ArgumentParser(
      formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument("--batch_size",
                        help="Batch Size",
                        type=int,
                        default=100)
    
    parser.add_argument("--epochs",
                        help="Number of runs",
                        type=int,
                        default=15)
    
    parser.add_argument("--train",
                        help="path to training data",
                        type=str,
                        default="")  
    
    parser.add_argument("--test",
                        help="path to test image",
                        type=str,
                        default="")
    
    parser.add_argument("--k",
                        help="number of k top images",
                        type=int,
                        default="5")
    
    params = parser.parse_args()
    
    data_dir = './geological_similarity_split/'
    device = torch.device('cuda')    
    transform_test = transforms.Compose([transforms.ToTensor(), transforms.Resize((32,32))])
    trainset_no_aug = ImageFolder(data_dir+'/set_train/', transform_test)
    trainloader_no_aug = DataLoader(trainset_no_aug, params.batch_size, num_workers=3, pin_memory=True)
    trainloader_no_aug = DeviceDataLoader(trainloader_no_aug, device)
    
    if params.train != '':
        if os.path.exists(params.train+'/set_train/') and os.path.exists(params.train + '/set_test/'):
            device = torch.device('cuda')    
            model = ResNet9(3, 6)
            model.to(device)
            train_network(model, data_dir = params.train)
            
            return
        else:
            raise Exception("Could not find training and/or testing data")
  
    if params.test != '': 
        if os.path.exists('model_weights.pth') and os.path.exists(params.test):
            model = ResNet9(3, 6) # we do not specify pretrained=True, i.e. do not load default weights
            model.load_state_dict(torch.load('model_weights.pth'))
            model.to(device)
            image = Image.open(params.test)
            show_sample_k(image, params.k)
            
            return
        else:
            raise Exception("Could not find test image")
            
        
if __name__ == "__main__":
    main()
        
      
    