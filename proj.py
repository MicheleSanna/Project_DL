from __future__ import print_function, division

import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
from torch.autograd import Variable
from torchsummary import summary
from torchvision.models import efficientnet_b0
from torchvision import datasets, models, transforms
from dataset import Dataset
import time
import os
#from efficientnet.model import EfficientNet

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
use_gpu = torch.cuda.is_available() 
num_classes = 2
blur = transforms.GaussianBlur(5, sigma=(0.01, 3))
batch_size=64
data_dir = "dataset/"
data_transform = transforms.Compose([
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(), # data augmentation
        ])





def train_model(model, dataloaders, criterion, optimizer, lr_scheduler, num_epochs=50, transform =None):
    train_loss = []
    since = time.time()
    for epoch in range(num_epochs):
        for phase in ['train', 'val']:
            size = len(dataloaders[phase].dataset)
            print('Phase: ' + phase + ' | batch_size: ' + str(batch_size) + ' | Learning rate: ' + str(optimizer.param_groups[0]['lr']))
            print('Epoch {}/{}'.format(epoch, num_epochs))
            print('-' * 10)

            running_loss = 0.0
            running_corrects = 0
            count = 0

            if phase == 'train':
                model.train(True)  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            for inputs, labels in dataloaders[phase]:
                if use_gpu:
                    inputs = inputs.to(device) 
                    labels = labels.to(device)
                else:
                    inputs, labels = inputs, labels
                            
                optimizer.zero_grad()
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1) 
                loss = criterion(outputs, labels)

                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                count += 1
                if count % 15 == 0 or outputs.size()[0] < batch_size:
                    print('Epoch:{}: loss:{:.3f}'.format(epoch, loss.item()) + " | count:" + str(count))
                    train_loss.append(loss.item())
                    print(str(count*batch_size) + "/" + str(size))
                    

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / size
            epoch_acc = running_corrects.double() / size
            #if phase == 'val':
            #    lr_scheduler.step(epoch_acc)
            print('Loss: {:.10f} Acc: {:.10f}'.format(
                epoch_loss, epoch_acc))

                
            if (phase == 'train'):
                save_dir = 'model'
                model_out_path = save_dir + "/" + "efficientnet" + '.pth'
                torch.save(model, model_out_path)
                print("Saved")

        time_elapsed = time.time() - since
        print('Epoch complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))

    return train_loss

#dataset = Dataset("dataset/", 'train')

#print("No transform")
print("Transform")
image_datasets = {x: Dataset("dataset/", x) for x in ['train', 'val']}
dataloaders_dict = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=False) for x in ['train', 'val']}
#['test'] = torch.utils.data.DataLoader(image_datasets['test'], batch_size=180, shuffle=True, num_workers=8, pin_memory=True)

model = torch.load('model/efficientnet_7.pth')
print("Device: :", device)
num_features = model.classifier[1].in_features     #extract fc layers features
model.classifier[1] = nn.Linear(num_features, num_classes)
criterion = nn.CrossEntropyLoss()

if use_gpu:
    model = model.cuda()
    criterion = criterion.cuda()
    
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))

train_model(model, dataloaders_dict, criterion, optimizer, torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=5, threshold=0.001, verbose=True), num_epochs=1, transform=data_transform)