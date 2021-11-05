import spconv
import torch.nn as nn
from .utils import subMVonvBlock, kaiming_init, toSparseInput

class Parser(nn.Module):

    def __init__(self):
        super(Parser, self).__init__()

        self.net = spconv.SparseSequential(
            subMVonvBlock( 1, 16, kernel_size=3),
            subMVonvBlock(16, 32, kernel_size=3),
            subMVonvBlock(32, 16, kernel_size=3),
            subMVonvBlock(16,  1, kernel_size=3)
        )

        self.toDense = spconv.ToDense()

        # Initialize weights
        self.net.apply(kaiming_init)


    def forward(self, inputTSDF):

        if not isinstance(inputTSDF, spconv.SparseConvTensor):  # TSDF input without fuser
            inputTSDF = toSparseInput(inputTSDF)
            inputTSDF.features = inputTSDF.features / 2 + 0.5   # Scale to [0, 1]

        res = self.net(inputTSDF)
        res = self.toDense(res) * 2 - 1     # Scale back to [-1, 1]

        return res


