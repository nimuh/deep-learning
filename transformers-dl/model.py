from transformers import GPT2Model
import torch
import torch.nn.functional as F


class ClassifierHead(torch.nn.Module):
    def __init__(self):
        super(ClassifierHead, self).__init__()

    def forward(self, x):
        pass


# Decoder module that takes the last hidden state of the GPT2 model 
# and reconstructs the original image like and autoencoder
class DecoderHead(torch.nn.Module):
    def __init__(self):
        super(DecoderHead, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 1, kernel_size=(3,3))
        self.pool1 = torch.nn.MaxPool2d((4,4), stride=(2, 25))
        self.upsample = torch.nn.Upsample(size=(28, 28))

    def forward(self, x):
        z = x.last_hidden_state
        h = z.resize(z.size(0), 1, z.size(1), z.size(2))
        h = self.conv1(h)
        h = self.pool1(h)
        h = F.relu(h)
        h = self.upsample(h)
        return h


class DecoderTransformer(torch.nn.Module):
    def __init__(self):
        super(DecoderTransformer, self).__init__()
        self.input_layer = torch.nn.Linear(16, 768)
        self.gpt = GPT2Model.from_pretrained('gpt2')
        self.gpt.eval()
        self.output_head = DecoderHead()

    def forward(self, x):
        h = self.input_layer(x)
        h = self.gpt(inputs_embeds=h)
        output = self.output_head(h)
        return output