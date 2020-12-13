import torch
from torch.autograd import Variable
import random
from torchvision.datasets import Cityscapes
from torchvision import transforms
from model import SegNet
from torch.utils.data import DataLoader
import wandb
import sys
import numpy as np
from sklearn.metrics import jaccard_score as jsc

IN_CHANNELS = 3
CLASSES = 34

gpu = False
monitor = False

if len(sys.argv) > 1 and sys.argv[1] == 'gpu':
    print("GPU: ", torch.cuda.is_available())
    gpu = torch.cuda.is_available()

if len(sys.argv) == 3 and sys.argv[2] == 'monitor':
    wandb.init(project='segnet')
    monitor = True

EPOCHS = 500
LR = 1e-4
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

tr_data = Cityscapes('./data', 
                     target_type=['semantic'], 
                     split='train', 
                     transform=in_t, 
                     target_transform=out_t,
                    )
v_data = Cityscapes('./data', 
                     target_type=['semantic'], 
                     split='val', 
                     transform=in_t, 
                     target_transform=out_t,
                    )

train_data = DataLoader(tr_data, batch_size=BATCH_SIZE, shuffle=True)
val_data = DataLoader(v_data, batch_size=BATCH_SIZE, shuffle=False)

# define model
model = SegNet(IN_CHANNELS, CLASSES)

if gpu:
    model.to(torch.device("cuda:0"))

if monitor:
    wandb.watch(model)

# optimizer and loss definition
criterion = torch.nn.CrossEntropyLoss().cuda()
optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum=0.9)

# training
for epoch in range(EPOCHS):

    tr_epoch_loss = 0
    tr_epoch_iou = 0
    tr_batch_count = 0
    val_batch_count = 0 
    val_epoch_loss = 0
    val_epoch_iou = 0

    print('-------- EPOCH {} -------'.format(epoch))

    # TRAINING
    for i, (inputs, targets) in enumerate(train_data, 0):

        labels = targets.squeeze(1).type(torch.int64)
        
        if gpu:
            inputs = inputs.cuda()
            labels = labels.cuda()

        logits, y_pred = model(inputs)

        batch_iou = jsc(
            labels.cpu().numpy().reshape(-1),
            torch.argmax(y_pred, dim=1).cpu().numpy().reshape(-1),
            average='macro',
        )

        tr_epoch_iou += batch_iou
        tr_batch_count += 1

        # Zero gradients, perform a backward pass, and update the weights.
        optimizer.zero_grad()
        tr_loss = criterion(logits, labels)
        tr_epoch_loss += tr_loss.item()
        print("----- BATCH: {} TRAIN LOSS: {}".format(i, tr_loss.item()))

        # backprop
        tr_loss.backward()
        optimizer.step()

    # VALIDATION
    for i, (inputs, targets) in enumerate(val_data, 0):

        labels = targets.squeeze(1).type(torch.int64)
        
        if gpu:
            inputs = inputs.cuda()
            labels = labels.cuda()

        logits, y_pred = model(inputs)

        batch_iou = jsc(
            labels.cpu().numpy().reshape(-1),
            torch.argmax(y_pred, dim=1).cpu().numpy().reshape(-1),
            average='macro',
        )

        val_epoch_iou += batch_iou
        
        val_batch_count += 1
        val_loss = criterion(logits, labels)
        val_epoch_loss += val_loss.item()
       
        print("----- BATCH: {} VAL LOSS: {}".format(i, val_loss.item()))

    # UPDATE GRAPHS FOR TRAINING AND VALIDATION
    TR_LOSS = tr_epoch_loss / len(train_data)
    TR_IOU = tr_epoch_iou / len(train_data)
    VAL_LOSS = val_epoch_loss / len(val_data)
    VAL_IOU = val_epoch_iou / len(val_data)

    if monitor:
            wandb.log({"training loss": TR_LOSS,
                       "training IoU": TR_IOU,
                       "validation loss": VAL_LOSS,
                       "validaton IoU": VAL_IOU,
                    })

    print('---------------------------')
    print("EPOCH: {} \
           TRAIN_LOSS: {} \
           TRAIN IoU {} \
           VAL LOSS: {} \
           VAL IoU: {} ".format(epoch,
                                TR_LOSS,
                                TR_IOU,
                                VAL_LOSS,
                                VAL_IOU,
                            ))
    print('---------------------------')

model_path = "./model_epochs{}_lr{}".format(EPOCHS, LR)
print("Saving Model...")
torch.save(model, model_path)
    
