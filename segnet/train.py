import torch
from torch.autograd import Variable
import random
from torchvision.datasets import Cityscapes
from torchvision import transforms
from model import SegNet
from torch.utils.data import DataLoader
import wandb
import sys

#wandb.init(project='segnet')
gpu = False
if len(sys.argv) > 1 and sys.argv[1] == 'gpu':
    print("GPU: ", torch.cuda.is_available())
    gpu = torch.cuda.is_available()

EPOCHS = 50
LR = 1e-4
print("----INIT TRAIN SCRIPT----")
print("=========================")
print("Train for {} EPOCHS with learning rate {:.3}".format(EPOCHS, LR))
#config = wandb.config
#config.learning_rate = LR


# Get data and define transformations
t = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    ])

data = Cityscapes('./data', target_type=['color'], transform=t, target_transform=t)

train_data = DataLoader(data, batch_size=16, shuffle=True)

# define model
model = SegNet(64, 64)
if gpu:
    model.to(torch.device("cuda:0"))

#wandb.watch(model)

# optimizer and loss definition
criterion = torch.nn.MSELoss().cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# training
for epoch in range(EPOCHS):
    epoch_loss = 0
    print('-------- EPOCH {} -------'.format(epoch))
    for i, (inputs, targets) in enumerate(train_data, 0):
        inputs = inputs.permute(0, 2, 3, 1)
        
        if gpu:
            inputs = inputs.cuda()
            #targets = targets.cuda()

        y_pred = model(inputs)

        # Compute and print loss
        loss = criterion(y_pred, inputs)
        print("BATCH: {} LOSS: {}".format(i, loss.item()))
        """
        wandb.log({"epoch": epoch, 
                   "loss": loss.item(),
                   })
        """
        # Zero gradients, perform a backward pass, and update the weights.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step
