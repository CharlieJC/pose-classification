#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms

#constants
num_epochs = 30

#class CustomImageDataset(Dataset):
#    def __init__(self, directory):
#        self.pose_dirs = os.listdir(directory)
#        self.images = []
#        for pose_id, pose_dir in enumerate(sorted(os.listdir(directory))):
#            full_pose_dir = directory + "/" + pose_dir
#            for img_file in os.listdir(full_pose_dir):
#                full_img_file = full_pose_dir + "/" + img_file
#                self.images.append((full_img_file, pose_id))
#        
#    def __len__(self):
#        return len(self.images)
#    
#    def __getitem__(self, idx):
#        img_file, pose_id = self.images[idx]
#        image = Image.open(img_file).convert('RGB')
#        label = pose_id
#        return image, label
#

#could replace classes with nn.Sequential
class ConvolutionalBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvolutionalBlock, self).__init__()
        self.conv = torch.nn.Conv2d(in_channels, out_channels, 3,2) 
        self.maxPool = torch.nn.MaxPool2d(2,2)
        # num_features = out_channels from conv2d
        self.batchNorm = torch.nn.BatchNorm2d(out_channels)
        self.selu = torch.nn.SELU()
    def forward(self, x):
        x = self.conv(x)
        x = self.maxPool(x)
        x = self.batchNorm(x)
        return self.selu(x)

class Attention(torch.nn.Module):
    def __init__(self, channels):
        super(Attention, self).__init__()
        # (output_size)??
        self.adaptAvgPool = torch.nn.AdaptiveAvgPool2d(channels)
        self.linear1 = torch.nn.Linear(channels, channels//16)
        self.sigmoid = torch.nn.Sigmoid()
        self.linear2 = torch.nn.Linear(channels//16, channels)
        self.relu = torch.nn.ReLU()
    def forward(self, x):
        x = self.adaptAvgPool(x)
        x = self.linear1(x)
        x = self.sigmoid(x)
        x = self.linear2(x)
        return self.relu(x)
        
branch = torch.nn.Sequential(ConvolutionalBlock(3, 24),
                     Attention(24),
                     ConvolutionalBlock(24,40),
                     ConvolutionalBlock(40, 64))

class SiameseNetwork(torch.nn.Module):
    def __init__(self, numClasses):
        super(SiameseNetwork,self).__init__()
        self.branch1 = branch
        self.branch2 = branch
        self.linear1 = torch.nn.Linear(64,numClasses)
        self.linear2 = torch.nn.Linear(64,numClasses*numClasses)
    
    def forward(self, x):
        branch1_out = self.branch1(x)
        branch2_out = self.branch2(x)
        linear1_out = self.linear1(branch1_out)
        linear2_out = self.linear2(branch1_out)
        linear3_out = self.linear2(branch2_out)
        return linear1_out, linear2_out, linear3_out
        
        
#https://pytorch.org/docs/stable/generated/torch.nn.CosineEmbeddingLoss.html
#https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html


#USE IMAGEFOLDER CLASS FOR LOADING IMAGES
 transform_img = transforms.Compose([
                 transforms.ToTensor(),
                 transforms.Normalize()
                 ])
 
 dataset_dir = "/workspace/data/yoga82/skeletons"
 
 dataset = torchvision.datasets.ImageFolder(
           root=dataset_dir, transform=transform_img
           )
 

 batch_size = 32 #check whats used in the paper
#clarify num_workers parameter
data_loader = DataLoader(dataset, batch_size,shuffle=True, num_workers=4)
model = SiameseNetwork(3)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
#for epoch in range(num_epochs):
#    for images,labels in data_loader:
#        y_linear1, y_linear2, y_linear3 = model(images)
#        cross_entropy_loss = torch.nn.CrossEntropyLoss()
#        cosine_embedding_loss = torch.nn.CosineEmbeddingLoss()
#                          
#        loss = cross_entropy_loss() + cosine_embedding_loss() #finish this, test on yoga 82 with yolov7-COCO
#       
#        loss.backward()
#    
#        optimizer.step()
#        optimizer.zero_grad()
#   

#https://towardsdatascience.com/how-to-calculate-the-mean-and-standard-deviation-normalizing-datasets-in-pytorch-704bd7d05f4c

def get_mean_and_std(dataloader):
    channels_sum, channels_squared_sum, num_batches = 0, 0, 0
    for data, _ in dataloader:
        # Mean over batch, height and width, but not over the channels
        channels_sum += torch.mean(data, dim=[0,2,3])
        channels_squared_sum += torch.mean(data**2, dim=[0,2,3])
        num_batches += 1
                            
    mean = channels_sum / num_batches

    # std = sqrt(E[X^2] - (E[X])^2)
    std = (channels_squared_sum / num_batches - mean ** 2) ** 0.5

    return mean, std


def main():
    mean, std = get_mean_and_std(data_loader)
    print("Mean:", mean)
    print("Std", std)


if __name__ == "__main__":
    main()

