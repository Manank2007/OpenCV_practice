import cv2 as cv
import numpy as np
import torch 
import torch.nn as nn
import torchvision
from torchvision.transforms import transforms
import matplotlib.pyplot as plt
import  torch.nn.functional as F

device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#Model parameters
num_epochs=4
num_batches=32
learning_rate=0.001


# dataset has PILImage images of range [0, 1]. 
# We transform them to Tensors of normalized range [-1, 1]
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


#Data loading :
train_data=torchvision.datasets.CIFAR10(root='./data',train=True,transform=transform,download=True)
test_data=torchvision.datasets.CIFAR10(root='./data',train=False,transform=transform,download=False)

test_loading=torch.utils.data.DataLoader(dataset=test_data,batch_size=num_batches,shuffle=False,)

train_loading=torch.utils.data.DataLoader(dataset=train_data,batch_size=num_batches,shuffle=True,)


classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


# get some random training images
dataiter = iter(train_loading)
images, labels = next(dataiter)

# show images
imshow(torchvision.utils.make_grid(images))

#   CNN building:
class ConNN(nn.Module):
    def __init__(self):
     super(ConNN,self).__init__()
     self.conl1=nn.Conv2d(3,6,5,1)
     self.con_pooling=nn.MaxPool2d(2,2)
     self.conl2=nn.Conv2d(6,16,5)
     self.fc1=nn.Linear(16*5*5,120)
     self.fc2=nn.Linear(120,104)
     self.fc3=nn.Linear(104,10)

    def forward(self,x):
     x=self.con_pooling(F.relu(self.conl1(x))) # >>> n,6,14,14
     x=self.con_pooling(F.relu(self.conl2(x))) # >>>> n,16,5,5
     x=x.view(-1,16*5*5) #to convert the final output from the CNN layer from 3 inputs to 2 inputs
     x=F.relu(self.fc1(x))
     x=F.relu(self.fc2(x))
     x=self.fc3(x) #No activation function here as its included in the cross entropy loss function 
     return x


model=ConNN().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
n_total_steps=len(train_loading)
#training loop
for epoch in range (num_epochs):
   for i,(image,label) in enumerate (train_loading):
       # origin shape: [4, 3, 32, 32] = 4, 3, 1024
        # input_layer: 3 input channels, 6 output channels, 5 kernel size

        image=image.to(device)
        label=label.to(device)


        #Forward pass
        output=model(image)
        loss=criterion(output,label)

        #Backward pass:
        optimizer.zero_grad()
        loss.backward()
        optimizer.step() #updates the weights and biases


        if (i+1) % 200== 0:
            print (f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{n_total_steps}], Loss: {loss.item():.4f}')

print('Finished Training')
PATH = './cnn.pth'
torch.save(model.state_dict(), PATH)
#testing loop
with torch.no_grad():
    n_correct = 0
    n_samples = 0
    n_class_correct = [0 for i in range(10)]
    n_class_samples = [0 for i in range(10)]
    for images, labels in test_loading:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        # max returns (value ,index)
        _, predicted = torch.max(outputs, 1)
        n_samples += labels.size(0)
        n_correct += (predicted == labels).sum().item()
        
        for i in range(num_batches):
            label = labels[i]
            pred = predicted[i]
            if (label == pred):
                n_class_correct[label] += 1
            n_class_samples[label] += 1

    acc = 100.0 * n_correct / n_samples
    print(f'Accuracy of the network: {acc} %')

    for i in range(10):
        acc = 100.0 * n_class_correct[i] / n_class_samples[i]
        print(f'Accuracy of {classes[i]}: {acc} %')








