import torch
from torch.autograd import Variable
import random
from torchvision.datasets import Cityscapes
from torchvision import transforms
from model import SegNet, DataLabels
from torch.utils.data import DataLoader
import wandb
import sys
import numpy as np
from sklearn.metrics import jaccard_score as jsc

# TODO
# MAP RGB VALUES IN MASK TO LABELS SO WE CAN TRAIN
# Write class that holds class information
# create method that maps a batch of masks to labels for training


IN_CHANNELS = 3
CLASSES = 6
gpu = False
monitor = False

color2label = {(0, 0, 0):      0,
               (128, 64, 128): 7,
               (244, 35, 232): 8,
               (250, 170, 30): 19,
               (220, 20, 60):  24,
               (0, 0, 142):    26,
              }

labeler = DataLabels(color2label)

if len(sys.argv) > 1 and sys.argv[1] == 'gpu':
    print("GPU: ", torch.cuda.is_available())
    gpu = torch.cuda.is_available()

if len(sys.argv) == 3 and sys.argv[2] == 'monitor':
    wandb.init(project='segnet')

EPOCHS = 500
LR = 1e-6
BATCH_SIZE = 16
print("----INIT TRAIN SCRIPT----")
print("=========================")
print("Train for {} EPOCHS with learning rate {:.3}".format(EPOCHS, LR))

if monitor:
    config = wandb.config
    config.learning_rate = LR


# Get data and define transformations
in_t = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        ])

out_t = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.functional.pil_to_tensor,
        ])

data = Cityscapes('./data', target_type=['color'], transform=in_t, target_transform=out_t)

train_data = DataLoader(data, batch_size=BATCH_SIZE, shuffle=True)

# define model
model = SegNet(IN_CHANNELS, CLASSES)

if gpu:
    model.to(torch.device("cuda:0"))

if monitor:
    wandb.watch(model)

# optimizer and loss definition
criterion = torch.nn.CrossEntropyLoss().cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# training
for epoch in range(EPOCHS):
    epoch_loss = 0
    print('-------- EPOCH {} -------'.format(epoch))
    batch_count = 0
    epoch_iou = 0
    for i, (inputs, targets) in enumerate(train_data, 0):

        labels = labeler.label_batch(targets)

        if gpu:
            inputs = inputs.cuda()
            labels = labels.cuda()

        logits, y_pred = model(inputs)
        
        # Compute and print loss
        loss = criterion(logits, labels)
        """
        batch_iou = jsc(
            y.cpu().numpy().reshape(-1),
            torch.argmax(y_pred, dim=1).cpu().numpy().reshape(-1),
            average='macro',
        )
        epoch_iou += batch_iou
        """
        
        batch_count += 1
        epoch_loss += loss.item()

        # Zero gradients, perform a backward pass, and update the weights.
        optimizer.zero_grad()
        loss = criterion(logits, labels)
        print("----- BATCH: {} LOSS: {}".format(i, loss.item()))

        if monitor:
            wandb.log({"loss": loss.item(),
                    "IoU": batch_iou,
                    })

        loss.backward()
        optimizer.step

    print('---------------------------')
    print("EPOCH: {} LOSS: {} IoU: {}".format(epoch, epoch_loss / batch_count, epoch_iou / batch_count))
    print('---------------------------')

model_path = "./model_epochs{}_lr{}".format(EPOCHS, LR)
print("Saving Model...")
torch.save(model, model_path)
#torch.load(model_path)
    
