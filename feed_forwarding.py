import torch
import torch.nn as nn
import torchvision
import matplotlib.pyplot as plt
import numpy as np
import torchvision.transforms as transforms


#device config :
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') #uses the gpu or cpu depending on the avalibility

#model specifiers:

batch_size=100
num_epochs=2
input_size=784   #28*28 pixels of an image
hidden_layers=100     #increase or decrease according to your use 
learning_rate=0.001
num_classes=10  # the digits are divided into 0-9 so 10 classes 

#MNIST dataset

train_dataset=torchvision.datasets.MNIST(root='./data',train=True,transform=transforms.ToTensor(),download=True,) #total training samples=600,000

test_dataset=torchvision.datasets.MNIST(train=False,transform=transforms.ToTensor(),root='./data') #total test samples=100,000

#Data  loader

test_loading=torch.utils.data.DataLoader(dataset=test_dataset,batch_size=batch_size,shuffle=True) #efficiently divides the data into batches and also allow different  mixing for better training of a model
#gives total tuples=(100,000/no of batches)

train_loading=torch.utils.data.DataLoader(dataset=train_dataset,batch_size=batch_size,shuffle=False)
 #gives total tuples=(600,000/no of batches )

examples=iter(test_loading)#converts the variable to a iterable 
example_data, example_target=next(examples) #unpacks the tuple from the examples into target and data and next()calls the first object of the tuples

#print (example_data.shape,example_target.shape)

for i in range(6):
    plt.subplot(2,3,i+1)
    plt.imshow(example_data[i][0], cmap='gray')
#plt.show()


#Neural network with one hidden layer :
class Neural_Network(nn.Module):  #nn.module acts as a managing framework we have to have a function ___init__
    def __init__(self,input_size,hidden_size,num_classes):
        super(Neural_Network,self).__init__()
        #The function of super().__init__() is to call the constructor of the parent class (nn.Module) and initialize its background machinery 
        # inside your custom class.
        self.input_size=input_size
        #first neural network layer :
        self.l1=nn.Linear(input_size,hidden_size)
        #activation function
        self.relu=nn.ReLU()
        #output layer
        self.l2=nn.Linear(hidden_size,num_classes)

        
    def forward(self,x):
            out=self.l1(x)
            out=self.relu(out)
            out=self.l2(out)
            #no activation function yet 
            return out       

model=Neural_Network(input_size,hidden_layers,num_classes)
model.to(device)   

#Loss and optimizer :
criterion=nn.CrossEntropyLoss()
optimizer=torch.optim.Adam(model.parameters(),lr=learning_rate)

#Training the model :

n_total_steps=len(train_loading)
for epoch in range (num_epochs):
     
    for i, (images,labels) in enumerate (train_loading):
      
      #loop over every epoch 
    #resizing the input from (100,1,28,28) to (784,100)
      images=images.reshape(-1,28*28).to(device)
      labels=labels.to(device)

    #forward pass :
      outputs= model(images) #images works as x (input)
      loss=criterion(outputs,labels) #labels works as y (expected outcome)

    #backward pass:
      optimizer.zero_grad()# initialiize the parameters(weights and biases to zero after every epoch)
      loss.backward()  #.backward is an inbuilt function that  stores the new calculated weights and biases
      optimizer.step()#The optimizer.step() function updates the neural network's weights and biases based on the gradients calculated during the backward pass.
      if (i+1) % 100 == 0:
            print (f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{n_total_steps}], Loss: {loss.item():.4f}')

#testing the model
#we dont want all the gradients used during training so we use 
with torch.no_grad():
    #Context-manager that disables gradient calculation
    n_correct = 0
    n_samples = 0
    for images, labels in test_loading:
        images = images.reshape(-1, 28*28).to(device)  #-1 gives the instruction to pytorch to automaticaaly calculate that specific dimension.
        labels = labels.to(device)
        outputs = model(images)
        # max returns (value ,index)
        value_probab, predicted = torch.max(outputs.data, 1) #value_probab gives the values of the probablity of an image belonging to a specific class 
        #predicted gives the class having the highest probablity in our case (predicted output)
        n_samples += labels.size(0)
        n_correct += (predicted == labels).sum().item() #.item() unpacks the tensor into a float or int >> only for 0D tensor 

    acc = 100.0 * n_correct / n_samples
    print(f'Accuracy of the network on the 10000 test images: {acc} %')
 