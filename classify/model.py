#!/usr/bin/env python
import torch
import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms

#constants
num_epochs = 30

#could replace classes with nn.Sequential
class ConvolutionalBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvolutionalBlock, self).__init__()
        self.seq = torch.nn.Sequential(torch.nn.Conv2d(in_channels, out_channels, kernel_size=3,stride=2),
                                        torch.nn.MaxPool2d(kernel_size=2, stride=2),
                                        torch.nn.BatchNorm2d(out_channels),
                                        torch.nn.SELU())
    def forward(self, x):
        return self.seq(x)

class Attention(torch.nn.Module):
    def __init__(self, channels):
        super(Attention, self).__init__()
        # (N,C,1,1)
        self.seq = torch.nn.Sequential(torch.nn.AdaptiveAvgPool2d(1),
                                        torch.nn.Flatten(),
                                        torch.nn.Linear(channels, channels//16),
                                        torch.nn.Sigmoid(),
                                        torch.nn.Linear(channels//16, channels),
                                        torch.nn.ReLU())

    def forward(self, x):
        y = self.seq(x)
        y = y.view(y.shape[0], -1, 1, 1) # reshape for broadcasting multiplication
        return x * y
        

class Branch(torch.nn.Module):
    def __init__(self):
        super(Branch,self).__init__()
        self.seq = torch.nn.Sequential(ConvolutionalBlock(3, 24),
                        Attention(24),
                        ConvolutionalBlock(24,40),
                        ConvolutionalBlock(40, 64))
    
    def forward(self, x):
        return self.seq(x)

class SiameseNetwork(torch.nn.Module):
    def __init__(self, numClasses):
        super(SiameseNetwork,self).__init__()
        self.branch1 = Branch()
        self.branch2 = Branch()
        self.linear1 = torch.nn.Linear(3*3*64,numClasses)
        # self.linear2 = torch.nn.Linear(3*3*64,numClasses*numClasses)
    
    def forward(self, x):
        branch1_out = self.branch1(x)
        # branch2_out = self.branch2(x)
        #flatten branch outputs
        branch1_out = branch1_out.view(branch1_out.size(0), -1)
        # branch2_out = branch2_out.view(branch2_out.size(0), -1)
        linear1_out = self.linear1(branch1_out)
        # linear2_out = self.linear2(branch1_out)
        # linear3_out = self.linear2(branch2_out)
        return linear1_out
        
        
#https://pytorch.org/docs/stable/generated/torch.nn.CosineEmbeddingLoss.html
#https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html

#constants
dataset_dir = "C:\\Users\\charl\\Desktop\\industrial project\\pose-classification\\data\\yoga82\\skeletons"
batch_size = 32 


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




#https://towardsdatascience.com/how-to-calculate-the-mean-and-standard-deviation-normalizing-datasets-in-pytorch-704bd7d05f4c



def main():
    dataset = torchvision.datasets.ImageFolder(
            root=dataset_dir, transform=transforms.ToTensor()
            )
    data_loader = DataLoader(dataset, batch_size,shuffle=True, num_workers=4)

    mean, std = get_mean_and_std(data_loader)

    #USE IMAGEFOLDER CLASS FOR LOADING IMAGES
    transform_img = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(mean, std)
                    ])
    
    normalized_dataset = torchvision.datasets.ImageFolder(
            root=dataset_dir, transform=transform_img
            )
    

    #check whats used in the paper
    #clarify num_workers parameter
    normalized_data_loader = DataLoader(normalized_dataset, batch_size,shuffle=True, num_workers=4)
    model = SiameseNetwork(3)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    losses = []
    for epoch in range(num_epochs):
        for images,labels in normalized_data_loader:
            # y_linear1, y_linear2, y_linear3 = model(images)
            y_linear1 = model(images)

            cross_entropy_loss = torch.nn.CrossEntropyLoss()
            # cosine_embedding_loss = torch.nn.CosineEmbeddingLoss()
                                
            # loss = cross_entropy_loss(y_linear1, labels) + cosine_embedding_loss(y_linear2,y_linear3) #finish this, test on yoga 82 with yolov7-COCO
            loss = cross_entropy_loss(y_linear1, labels)
            loss.backward()
        
            optimizer.step()
            optimizer.zero_grad()

            losses.append(loss)
  
  


if __name__ == "__main__":
    main()