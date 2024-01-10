import glob
import random
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torchsummary import summary
from torchvision.models import efficientnet_b0
from torchvision import datasets, models, transforms
import time
import os
#from efficientnet.model import EfficientNet
from torch.utils.data import Dataset, DataLoader

class Dataset(Dataset):
    def __init__(self, data_dir, mode, testing_models = None):
        self.data_transform = transforms.Compose([
                            transforms.RandomCrop(224),
                            transforms.RandomHorizontalFlip(), # data augmentation
                            ])
        file_list = []
        self.data = []
        self.imgs_path = data_dir + mode + "/"
        #if transform is not None:
        #    self.transform = transform
        #else:
        #    self.transform = transforms.Resize((224, 224))
        
        if (mode == 'train' or mode == 'val'):
            folder_list = glob.glob(self.imgs_path + "*")
            for folder in folder_list:
                file_list.append(glob.glob(folder + "/*"))
            print(file_list)
            file_list = [item for sublist in file_list for item in sublist]
            
        if (mode == 'test'):
            if (testing_models == None):
                testing_models = ['biggan','crn','cyclegan','deepfake','gaugan','imle','progan','san','seeingdark','stargan','stylegan','stylegan2','wichfaceisreal']
            
            for testing_model in testing_models:
                folder_list = glob.glob(self.imgs_path + testing_model + "/*")
                for folder in folder_list:
                    if (len(glob.glob(folder + "/*.png")) > 0):
                        file_list.append(folder)
                    else:
                        file_list = file_list + (glob.glob(folder + "/*"))
                        
        for class_path in file_list:
            class_name = class_path.split("/")[-1]
            for img_path in glob.glob(class_path + "/*.png"):
                self.data.append([img_path, class_name])
        self.class_map = {"0_real" : 0, "1_fake": 1}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, class_name = self.data[idx]
        img = cv2.imread(img_path)
        #img = cv2.resize(img, (224,224))
        x = random.randint(30,170)
        #DATA AUGMENT. commentare se si sta facendo la run senza data augm
        if x < 100:
            _, img = cv2.imencode('.jpeg', img, [int(cv2.IMWRITE_JPEG_QUALITY), x])
            img = cv2.imdecode(img, cv2.IMREAD_COLOR)

        img_tensor = torch.from_numpy(img)
        img_tensor = img_tensor.permute(2, 0, 1)
        img_tensor = img_tensor.float()
        img_tensor = self.data_transform(img_tensor)
        if bool(random.getrandbits(1)):
            #print("c")
            img_tensor = blur(img_tensor)

        class_id = self.class_map[class_name]
        class_id = torch.tensor(class_id)
        return img_tensor, class_id
    

        
    
