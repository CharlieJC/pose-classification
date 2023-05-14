#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader

#constants
num_epochs = 30

class CustomImageDataset(Dataset):
    def __init__(self, directory):
        self.directory = directory
        self.images = os.listdir(directory)
        
    def __len__(self):
        return len(self.img_list)
    
    def __getitem__(self, idx):
        path = os.path.join(self.directory, self.images[idx])
        image = Image.open(directory).convert('RGB')
        label = 1 # load label based on dataset structure
        return image,label


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
        self.adaptAvgPool = torch.nn.AdaptiveAvgPool2d()
        self.linear1 = torch.nn.Linear(channels, channels/16)
        self.sigmoid = torch.nn.Sigmoid()
        self.linear2 = torch.nn.Linear(channels/16, channels)
        self.relu = torch.nn.ReLU()
    def forward(self, x):
        x = self.adaptAvgPool(x)
        x = self.linear1(x)
        x = self.sigmoid(x)
        x = self.linear2(x)
        return self.relu(x)
        
branch = nn.Sequential(ConvolutionalBlock(3, 24),
                      AttentionBlock(24),
                     ConvolutionalBlock(24,40)
                     ConvolutionalBlock(40, 64)

class SiameseNetwork(torch.nn.Module):
    def __init__(self, numClasses):
        self.branch1 = branch
        self.branch2 = branch
        self.linear1 = self.nn.Linear(64,numClasses)
        self.linear2 = self.nn.Linear(64,numClasses*numClasses)
    
    def forward(self, x):
        branch1_out = self.branch1(x)
        branch2_out = self.branch2(x)
        linear1_out = self.linear1(branch1_out)
        linear2_out = self.linear2(branch1_out)
        linear3_out = self.linear2(branch2_out)
        return linear1_out, linear2_out, linear3_out
        
        
#https://pytorch.org/docs/stable/generated/torch.nn.CosineEmbeddingLoss.html
#https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html

dataset_dir = "path"
                       
dataset = CustomImageDataset(directory=dataset_dir)
batch_size = 32 #check whats used in the paper
#clarify num_workers parameter
data_loader = DataLoader(dataset, batch_size,shuffle=True, num_workers=4)
model = SiameseNetwork(3)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
for epoch in range(num_epochs):
    for images,labels in data_loader:
        y_linear1, y_linear2, y_linear3, model(images)
        cross_entropy_loss = torch.nn.CrossEntropyLoss()
        cosine_embedding_loss = torch.nn.CosineEmbeddingLoss()
                          
        loss = cross_entropy_loss() + cosine_embedding_loss() #finish this, test on yoga 82 with yolov7-COCO
       
        loss.backward()
    
        optimizer.step()
        optimizer.zero_grad()
   
    
