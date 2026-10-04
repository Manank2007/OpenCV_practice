import torch 
import torchvision
from torch.utils.data import DataLoader, Dataset
import math
import numpy as np

class CustomDataset(Dataset):
    def __init__(self):
        #data loading
        xy=np.loadtxt("C:\\Users\\VICTUS\\Downloads\\wine.csv",delimiter=',',dtype=np.float32,skiprows=1)
        self.x=torch.from_numpy(xy[:,1:])
        self.y=torch.from_numpy(xy[:,[0]])
        self.n_samples= xy.shape[0]
        
        
    def __getitem__(self, index):
        #dataset indexing 
        return self.x[index],self.y[index]
    
    def __len__(self):
       return self.n_samples



dataset=CustomDataset()
train_loader=DataLoader(dataset=dataset,batch_size=4,shuffle=True,num_workers=0)
     #gives a tuple of inputs, labels for each batch so if batch size is 4 the number of tupels would be 4 

    #Dummy training loop 
num_epochs=2
total_samples=len(dataset)
batches=4
n_iterations=math.ceil(total_samples/batches) #math.ceil just roundoffs the values from divison 

for epoch in range(num_epochs): #loop over the number of epochs (here 2)
        for i ,(inputs, labels) in enumerate(train_loader):  #unpack the train_loader which gives the output as tuples of inputs and labels into variables named inputs and labels where both inputs and label have a size of (total samples/batch size ) here 45

            if (i+1) % 5 ==0:  #at every 5th step print the values given  below 
                   print(f'Epoch: {epoch+1}/{num_epochs}, Step {i+1}/{n_iterations}| Inputs {inputs.shape} | Labels {labels.shape}')
                    