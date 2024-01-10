from __future__ import print_function, division
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import torch.optim.lr_scheduler as lr_scheduler
import torchvision
from torchvision import transforms
from torchvision import datasets, models, transforms
from sklearn.metrics import average_precision_score
from torchsummary import summary
import numpy as np
import random
from PIL import ImageFilter
from PIL import Image, ImageFile
# import matplotlib.pyplot as plt
import cv2
import glob
import time
import os
import copy

noaug_acc = np.array([.67225, .7125, .5625, .9975, 1, .835253, .859289, .500235, .500392, .769406, .675, .552451])
aug_acc = np.array([.73, .8439, .7309, 1, .945473, .913434, .882949, .761752, .768881, .599045, .697222, .53469])
noaug_ap = np.array([.6248, .8291, .5515, .9999, 1, .9566, .994, .5894, .5911, .8284, .5076, .9243])
aug_ap = np.array([.8315, .9181, .6571, .99, 1, .996, .9994, .785, .8858, .437, .9094, .942])

# Detect if we have a GPU available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

inizio = time.time()
now_str = time.strftime("%b %d %Y %H:%M:%S", time.gmtime(inizio))


ImageFile.LOAD_TRUNCATED_IMAGES = True

#inizio = time.time()
print('start running')

"""## Useful functions and classes"""

# Testing
def testing(model, dataloader, criterion, optimizer, testset):
  start_time = time.time()
    
  model.eval()   # Set model to evaluate mode
  running_loss = 0.
  running_corrects = 0
  y_true = []
  y_score = []
    
  # load a batch data of images
  for i, (inputs, labels) in enumerate(dataloader):
      inputs = inputs.to(device)
      labels = labels.to(device) 
      #print(inputs)
      # forward inputs and get output
      optimizer.zero_grad()
      outputs = model(inputs)
      _, preds = torch.max(outputs, 1)
      loss = criterion(outputs, labels)

      running_loss += loss.item() * inputs.size(0)
      running_corrects += torch.sum(preds == labels.data)

      y_true.extend(labels.tolist())
      y_score.extend(torch.softmax(outputs, dim=1)[:, 1].tolist())
    
  loss = running_loss / len(dataloader.dataset)
  acc = running_corrects / len(dataloader.dataset) * 100.
  prec = average_precision_score(y_true, y_score)

  print(testset,' Loss: {:.4f} Acc: {:.4f}% AP: {:.4f}'.format(loss, acc, prec))
  print()

  final_time = time.time() - start_time
  print('Training complete in {:.0f}m {:.0f}s'.format(final_time // 60, final_time % 60))



"""## Upload the model"""

print("Model upload:\n")
tempo_trascorso = time.time() - inizio
print('Inizio fase: {:.0f}m {:.0f}s'.format(tempo_trascorso // 60, tempo_trascorso % 60))
print()

model = torch.load('model/bestmodel_no_aug.pth')

# Send the model to GPU
model = model.to(device)

# Set loss function
criterion = nn.CrossEntropyLoss()

# Set optimizer
#optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))

class myGaussianBlur:
    def __init__(self, radius=2):
        self.radius = radius

    def __call__(self, img):
        return img.filter(ImageFilter.GaussianBlur(radius=self.radius))

class JpegWithProbability:
    def __init__(self, probability=0.5, quality=95):
        self.probability = probability
        self.quality = quality

    def __call__(self, img):
        if random.random() < self.probability:
            img = np.array(img)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), self.quality]
            _, encimg = cv2.imencode('.jpg', img, encode_param)
            img = cv2.imdecode(encimg, cv2.IMREAD_COLOR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)

        return img

"""## Dataset preparation"""

print("Dataset preparation:\n")
tempo_trascorso = time.time() - inizio
print('Inizio fase: {:.0f}m {:.0f}s'.format(tempo_trascorso // 60, tempo_trascorso % 60))
print()

data_transforms = transforms.Compose([
        transforms.RandomCrop(224),
        #JpegWithProbability(),
        #transforms.RandomHorizontalFlip(),
        #transforms.RandomApply([myGaussianBlur(radius=2)], p=0.5),
        transforms.ToTensor(),
        ])

# create the dataset
data_dir = "dataset/test/"
#data_dir = "/u/dssc/ceciza/DL_project/imagefolder/"
#test_list = ['biggan', 'deepfake', 'gaugan', 'san', 'stargan']
#test_list = ['whichfaceisreal']
test_list = ['stylegan3']
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), transform=data_transforms) for x in test_list}

# Create dataloaders
dataloaders_test_dict = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=64, shuffle=False, num_workers=4) for x in test_list}


## Testing phase

print("Testing phase:\n")
tempo_trascorso = time.time() - inizio
print('Inizio fase: {:.0f}m {:.0f}s'.format(tempo_trascorso // 60, tempo_trascorso % 60))
print()

for cnn in test_list:
  print("------------------------------------")
  print("Model: ", cnn)
  testing(model,dataloaders_test_dict[cnn],criterion, optimizer,cnn)

tempo_trascorso = time.time() - inizio
print('Fine script: {:.0f}m {:.0f}s'.format(tempo_trascorso // 60, tempo_trascorso % 60))
