from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor, Resize
from torch import flatten
from transformers import GPT2Model
import torch
import model
from tqdm import tqdm
import wandb

import warnings

warnings.filterwarnings("ignore")
wandb.init(project="transformers")

transform_pipeline = Compose([ToTensor(), Resize(size=(28, 28))])
data = MNIST(root='mnist', transform=transform_pipeline, download=True)
data_loader = DataLoader(data, batch_size=32, shuffle=True)


epochs = 30
lr = 1e-6

#decoder = model.DecoderTransformer()
decoder = model.CNN()
opt = torch.optim.Adam(decoder.parameters(), lr=lr)
loss_fn = torch.nn.MSELoss()


for epoch in range(epochs):

    epoch_loss = 0
    pbar = tqdm(total=len(data_loader))
    pbar.set_description("Epoch: {}".format(epoch))

    for _, (x, y) in enumerate(data_loader):

        opt.zero_grad()

        """
        # remove channel dimension
        inputs = flatten(x, start_dim=1, end_dim=2)

        # create 4x4 patches on each image
        inputs = flatten(inputs.unfold(1, 4, 4).unfold(2, 4, 4), 
                        start_dim=1, 
                        end_dim=2,
                        )

        # flatten patches 
        inputs = flatten(inputs, start_dim=2)
        """
        preds = decoder(x)

        loss = loss_fn(x, preds)
        epoch_loss += loss.item()

        loss.backward()
        opt.step()
        pbar.update(1)
        #sleep(0.5)

    pbar.close()
    print("LOSS: {}".format(epoch_loss / len(data_loader)))
    wandb.log({"CNN MSE Loss": epoch_loss/len(data_loader)})


