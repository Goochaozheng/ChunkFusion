import spconv
import torch
import torch.nn as nn

class FusionNet(torch.nn.Module):

    def __init__(self):
        super(FusionNet, self).__init__()

        self.parser = spconv.SparseSequential(
            self._subMVonvBlock(1 , 16, kernel_size=3),
            self._subMVonvBlock(16, 32, kernel_size=3),
            self._subMVonvBlock(32, 16, kernel_size=3),
            self._subMVonvBlock(16,  1, kernel_size=3)
        )

        self.fuser = spconv.SparseSequential(
            self._subMVonvBlock( 4, 16, kernel_size=3),
            self._subMVonvBlock(16, 32, kernel_size=3),
            self._subMVonvBlock(32, 16, kernel_size=3),
            self._subMVonvBlock(16,  1, kernel_size=3)
        )

        self.confidence = spconv.SparseSequential(
            self._subMVonvBlock(  1, 16, kernel_size=3),
            self._subMVonvBlock( 16, 16, kernel_size=3),
            self._subMVonvBlock( 16,  1, kernel_size=3)
        )

        self.toDense = spconv.ToDense()

        # Initialize weights
        self.parser.apply(self.weights_init)
        self.fuser.apply(self.weights_init)


    def _residualBlock(self, channels, kernel_size=3):
        return spconv.SparseSequential(
            spconv.ConcatTable()
                .add(spconv.Identity())
                .add(spconv.SparseSequential(
                    nn.BatchNorm1d(channels),
                    spconv.SubMConv3d(channels, channels, kernel_size=kernel_size),
                    nn.Sigmoid()
                )),
            spconv.JoinTable()
        )


    def _subMVonvBlock(self, inChannels, outChannels, kernel_size=3, indiceKey=None):
        return spconv.SparseSequential(
            nn.BatchNorm1d(inChannels),
            spconv.SubMConv3d(inChannels, outChannels, kernel_size=kernel_size, indice_key=indiceKey),
            nn.Sigmoid()
        )


    def weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            torch.nn.init.kaiming_uniform_(m.weight)  


    def toSparseInput(self, inputTSDF):

        # Construct Sparse Tensor
        inputTSDF = inputTSDF.permute(0,2,3,4,1)
        sparseMask = torch.any(torch.abs(inputTSDF) < 1, dim=4)
        batchSize = len(inputTSDF)
        spatialShape = inputTSDF.shape[1:-1]
        sparseIndice = sparseMask.to_sparse(inputTSDF.ndim-1).indices().permute(1, 0).contiguous().int()
        sparseValue = inputTSDF[sparseMask]
        
        inputData_sparse = spconv.SparseConvTensor(features=sparseValue, indices=sparseIndice, spatial_shape=spatialShape, batch_size=batchSize)

        return inputData_sparse



    def forward(self, inputTSDF, localTSDF=None):

        if localTSDF is None:
            inputTSDF_sparse = self.toSparseInput(inputTSDF)
            fuseTSDF_sparse = self.parser(inputTSDF_sparse)

        else:

            localTSDF_sparse = self.toSparseInput(localTSDF)
            inputTSDF_sparse = self.toSparseInput(inputTSDF)
            
            inputConfidence = self.confidence(inputTSDF_sparse)
            localConfidence = self.confidence(localTSDF_sparse)

            fuseTSDF_sparse = self.sparseFuse(inputTSDF_sparse, inputConfidence, localTSDF_sparse, localConfidence)

            fuseTSDF_sparse = self.parser(fuseTSDF_sparse)

        fuseTSDF = self.toDense(fuseTSDF_sparse) * 2 - 1

        return fuseTSDF


