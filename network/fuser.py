import spconv
import torch
from torch import nn
from .utils import subMVonvBlock, kaiming_init, toSparseInput

class Fuser(nn.Module):

    def __init__(self):
        super(Fuser, self).__init__()

        self.net = spconv.SparseSequential(
            subMVonvBlock(  2, 16, kernel_size=3),
            subMVonvBlock( 16, 32, kernel_size=3),
            subMVonvBlock( 32, 16, kernel_size=3),
            subMVonvBlock( 16,  1, kernel_size=3)
        )

        self.toDense = spconv.ToDense()

        # Initialize weights
        self.net.apply(kaiming_init)

    def forward(self, inputTSDF, localTSDF):

        res = toSparseInput(torch.cat((inputTSDF, localTSDF), dim=1))
        res = self.net(res)

        return res


