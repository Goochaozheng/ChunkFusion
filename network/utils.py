import spconv
import torch
from torch import nn

def residualBlock(channels, kernel_size=3):
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


def subMVonvBlock(inChannels, outChannels, kernel_size=3, indiceKey=None):
    return spconv.SparseSequential(
        nn.BatchNorm1d(inChannels),
        spconv.SubMConv3d(inChannels, outChannels, kernel_size=kernel_size, indice_key=indiceKey),
        nn.Sigmoid()
    )


def convBlock(inChannels, outChannels, kernel_size=3):
    return nn.Sequential(
        nn.BatchNorm3d(inChannels),
        nn.Conv3d(inChannels, outChannels, kernel_size=kernel_size, stride=1, padding=1),
        nn.LeakyReLU()
    )


def kaiming_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.kaiming_uniform_(m.weight)  


def toSparseInput(inputTSDF):

    # Construct Sparse Tensor
    inputTSDF = inputTSDF.permute(0,2,3,4,1)
    sparseMask = torch.any(torch.abs(inputTSDF) < 1, dim=4)
    batchSize = len(inputTSDF)
    spatialShape = inputTSDF.shape[1:-1]
    sparseIndice = sparseMask.to_sparse(inputTSDF.ndim-1).indices().permute(1, 0).contiguous().int()
    sparseValue = inputTSDF[sparseMask]
    
    inputData_sparse = spconv.SparseConvTensor(features=sparseValue, indices=sparseIndice, spatial_shape=spatialShape, batch_size=batchSize)

    return inputData_sparse


def sparseFuse(inputSparseTSDF, oldSparseTSDF, inputMask, oldMask):

    # fuseTSDF = torch.cat((self.toDense(inputSparseTSDF), self.toDense(oldSparseTSDF)), dim=1)
    oldTSDF = spconv.ToDense(oldSparseTSDF).permute(0,2,3,4,1)
    inputTSDF = spconv.ToDense(inputSparseTSDF).permute(0,2,3,4,1)

    # oldTSDF[inputMask] = (oldTSDF[inputMask] * oldWeight[inputMask] + inputTSDF[inputMask] * inputWeight[inputMask]) / (oldWeight[inputMask] + inputWeight[inputMask])

    batchSize = inputSparseTSDF.batch_size
    spatialShape = inputSparseTSDF.spatial_shape
    fuseMask = torch.logical_or(inputMask, oldMask)
    sparseIndice = fuseMask.to_sparse(oldTSDF.ndim-1).indices().permute(1, 0).contiguous().int()
    sparseValue = oldTSDF[fuseMask]

    return spconv.SparseConvTensor(features=sparseValue, indices=sparseIndice, spatial_shape=spatialShape, batch_size=batchSize)
