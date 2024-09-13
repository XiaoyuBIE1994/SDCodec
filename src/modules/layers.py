import torch
from torch import nn
from torch.nn import TransformerEncoderLayer
from torch.nn.utils.parametrizations import weight_norm
from torch.distributions import Categorical

def WNConv1d(*args, **kwargs):
    act = kwargs.pop("act", False)
    conv = weight_norm(nn.Conv1d(*args, **kwargs))
    if act:
        return nn.Sequential(conv, nn.LeakyReLU(0.1))
    else:
        return conv


def WNConvTranspose1d(*args, **kwargs):
    return weight_norm(nn.ConvTranspose1d(*args, **kwargs))



def WNConv2d(*args, **kwargs):
    act = kwargs.pop("act", False)
    conv = weight_norm(nn.Conv2d(*args, **kwargs))
    if act:
        return nn.Sequential(conv, nn.LeakyReLU(0.1))
    else:
        return conv
    
# Scripting this brings model speed up 1.4x
@torch.jit.script
def snake(x, alpha):
    shape = x.shape
    x = x.reshape(shape[0], shape[1], -1)
    x = x + (alpha + 1e-9).reciprocal() * torch.sin(alpha * x).pow(2)
    x = x.reshape(shape)
    return x

class Snake1d(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1, channels, 1))

    def forward(self, x):
        return snake(x, self.alpha)
    

class TransformerSentenceEncoderLayer(nn.Module):
    """
    Stacks of TransformerEncoderlayer used for conv layer output
    """
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 norm_first=False, bias=True, num_layers=1):
        super().__init__()
        layers = [TransformerEncoderLayer(d_model, nhead, dim_feedforward=dim_feedforward, dropout=0.1,
                                                   batch_first=False, norm_first=norm_first, bias=bias)
                for _ in range(num_layers)]
        self.blocks = nn.Sequential(*layers)
        
    def forward(self, x):
        assert len(x.shape) == 3
        x = x.permute(2, 0, 1) # (bs, C, L) -> (L, bs, C)
        x = self.blocks(x)
        x = x.permute(1, 2, 0) # (L, bs, C) -> (bs, C, L)
        return x


class SLSTM(nn.Module):
    """
    LSTM without worrying about the hidden state, nor the layout of the data.
    Expects input as convolutional layout.
    Modified from the EnCodec: https://github.com/facebookresearch/encodec/blob/main/encodec/modules/lstm.py
    """
    def __init__(self, dimension: int, num_layers: int = 2, bidirectional: bool = True, skip: bool = True):
        super().__init__()
        self.skip = skip
        self.bidirectional = bidirectional
        self.lstm = nn.LSTM(dimension, dimension, num_layers, bidirectional=self.bidirectional)
        if bidirectional:
            self.linear_ = nn.Linear(dimension*2, dimension)

    def forward(self, x):
        x = x.permute(2, 0, 1)
        y, _ = self.lstm(x)
        if self.bidirectional:
            y = self.linear_(y)
        if self.skip:
            y = y + x
        y = y.permute(1, 2, 0)
        return y
    


class Jitter(nn.Module):
    """
    Shuffule the input by randomly swapping with neighborhood
    Modified from the SQ-VAE speech: https://github.com/sony/sqvae/blob/main/speech/model.py#L76
    Args:
    p: probability to shuffle code
    size: kernel size to shuffle code
    """
    def __init__(self, p, size=3):
        super().__init__()
        self.p = p
        self.size = size
        prob = torch.ones(size) * p / (size - 1)
        prob[size//2] = 1 - p
        self.register_buffer("prob", prob)

    def forward(self, x):
        if not self.training or self.p == 0.0:
            return x
        else:
            batch_size, dim, T = x.size()

            dist = Categorical(probs=self.prob)
            index = dist.sample(torch.Size([batch_size, T])) - len(self.prob)//2
            index += torch.arange(T, device=x.device)
            index.clamp_(0, T-1)
            x = torch.gather(x, -1, index.unsqueeze(1).expand(-1, dim, -1))

        return x
    

if __name__ == "__main__":

    jit = Jitter(p=0.8, size=5)
    print(jit.prob)

    x = torch.arange(20).reshape(1,1,20)
    x = torch.cat((x, x+100), dim=0)
    print(x)
    print(jit(x))