'''
import json
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm
import time

BATCH_SIZE = 128
NUM_EPOCHS = 10

with open("./board_cases.json") as f:
    board_json = json.load(f)

print(board_json)

OUT_SHAPE = (4,4)
CAND = 16
map_table = {2**i: i for i in range(1,CAND)}
map_table[0] = 0

def grid_ohe(arr):
    ret = np.zeros(shape = OUT_SHAPE + (CAND,), dtype = bool)
    for r in range(OUT_SHAPE[0]):
        for c in range(OUT_SHAPE[1]):
            ret[r,c,map_table[arr[r,c]]] = 1
    return ret

# preprocessing
normalize = transforms.Normalize(mean=[.5], std=[.5])
transform = transforms.Compose([transforms.ToTensor(), normalize])

# download and load the data
train_dataset = torchvision.daatasets.MNIST(root='./mnist/', train=True, transform=transform, download=True)

# encapsulate them into dataloader form
train_loader = data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
train_x = torch.unsqueeze(train_dataset.train_data, dim=1).type(torch.FloatTensor)[:2000]/255.
train_y = train_dataset.train_labels[:2000]

class SimpleNet(nn.Module):
# TODO:define model
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=5,
                stride=1,
                padding=2,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2), 
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.out = nn.Linear(32 * 7 * 7, 10)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        output = self.out(x)
        return output, x
    
model = SimpleNet()

# define loss function and optimiter
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# train and evaluate
for epoch in range(NUM_EPOCHS):
    for step, (b_x, b_y) in enumerate(train_loader):
    # for images, labels in tqdm(train_loader):
        # forward + backward + optimize
        output = model(b_x)[0]                 
        loss = criterion(output, b_y)     
        optimizer.zero_grad() 
        loss.backward()        
        optimizer.step()
            
# evaluate
# calculate the accuracy using traning and testing dataset
train_output, last_layer = model(train_x)
pred_ty = torch.max(train_output, 1)[1].data.squeeze().numpy()
train_accuracy = float((pred_ty == train_y.data.numpy()).astype(int).sum()) / float(train_y.size(0))
'''

'''
Guide = nametuple('Guide',('state','action'))

class Guides:

    def _init_(self,capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self,*args):
        "Saves a transition."
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Guide(*args)
        self.position = (self.position+1)%self.capacity
    
    def sample(self,batch_size):
        return random.sample(self.memory,batch_size)

    def ready(self,batch_size):
        return len(self.memory) >= batch.size

    def _len_(self):
        return len(self.memory)
    '''