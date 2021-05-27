from numpy.core.fromnumeric import shape
import torch
import numpy as np
import pickle
import image_processing


from image_processing import read_path, path_name, images, labels, resize_image

import torch.nn.functional as functional
from torch.nn.modules.loss import CrossEntropyLoss
from torch.optim import optimizer

from torchvision import transforms
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image


hana_images, hana_labels = read_path(path_name)


custom_transforms = transforms.Compose(
    [transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

batch_size = 5
classes = ('Hana', 'Ben', 'Hammond', 'Ayoan')

class FacialDatabase(Dataset):
    def __init__(self, features, labels, transform=None):
        self.features = features
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        x = self.features[idx]
        y = self.labels[idx]

        if self.transform:
            x = self.transform(x)
            y = y

        return x, y

hana_trainset = FacialDatabase(hana_images, hana_labels, custom_transforms)
hana_trainloader = DataLoader(hana_trainset, batch_size, shuffle=True)


#create a CNN module
class CNN(nn.Module):
    def __init__(self):
        self.output_size = 4 #Ben, Hana, Hammond, Ayoan
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.ful_con1 = nn.Linear(16 * 5 * 5, 120)
        self.ful_con2 = nn.Linear(120, 84)
        self.ful_con3 = nn.Linear(84, self.output_size)
    
    def forward(self, x):
        x = self.pool(functional.relu(self.conv1(x)))
        x = self.pool(functional.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = functional.relu(self.ful_con1(x))
        x = functional.relu(self.ful_con2(x))
        x = self.ful_con3(x)
        return x
cnn = CNN()


criterion = nn.CrossEntropyLoss()
optimize = optim.SGD(cnn.parameters(), lr = 1e-3, momentum = 0.9)

#train 
for epoch in range(5):

    running_loss = 0.0
    for i, data in enumerate(hana_trainloader, 0):
        inputs, labels = data
        optimize.zero_grad()
        outputs = cnn(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimize.step()
        running_loss += loss.item()

        if i % 50 == 49:
            print("Epoch: %d, %5d loss: %.5f" % (epoch + 1, i + 1, running_loss / 5))
            running_loss = 0.0

torch.save(cnn, 'facial_cnn.pt')


def cnn_output(image):
    image = resize_image(image)
    image = image.reshape((1, 3, 32, 32))
    image = image.astype('float32')
    test_dataset = FacialDatabase(image, 'Hana')
    test_trainloader = DataLoader(test_dataset)
    
    
    dataiter = iter(test_trainloader)
    img, lbl = dataiter.next()
    output = cnn(img)

    sm = nn.Softmax(dim = 1)
    sm_output = sm(output)

    probs, index = torch.max(sm_output, dim =1)
    return probs, index

