import torch
import torch.nn as nn
import numpy as np

class FusionLoss(nn.Module):

    def __init__(self, w_l1=1., w_mse=1., w_sign=1., w_grad=1., reduction="mean"):
        super().__init__()

        self.reduction = reduction

        self.w_1 = w_l1
        self.w_2 = w_mse
        self.w_3 = w_sign
        self.w_4 = w_grad

        self.l1 = nn.L1Loss(reduction='none')
        self.l2 = nn.MSELoss(reduction='none')
        self.sign = SignLoss(reduction='none')
        self.grad = GradLoss(reduction='none')


    def forward(self, mask, input, target):

        maskedInput = input[mask]
        maskedTarget = target[mask]

        if self.w_1:
            l1_loss = self.l1(maskedInput, maskedTarget)
        else:
            l1_loss = torch.zeros_like(maskedInput)

        if self.w_2:
            l2_loss = self.l2(maskedInput, maskedTarget)
        else:
            l2_loss = torch.zeros_like(maskedInput)       

        if self.w_3:
            sign_loss = self.sign(maskedInput, maskedTarget)
        else:
            sign_loss = torch.zeros_like(maskedInput)       

        if self.w_4:
            grad_loss = self.grad(mask, input, target)
        else:
            grad_loss = torch.zeros_like(maskedInput)    

        if self.reduction == "mean":
            l1_loss = l1_loss.mean()
            l2_loss = l2_loss.mean()
            sign_loss = sign_loss.mean()
            grad_loss = grad_loss.mean()
            
        elif self.reduction == "sum":
            l1_loss = l1_loss.sum()
            l2_loss = l2_loss.sum()
            sign_loss = sign_loss.sum()
            grad_loss = grad_loss.sum()

        loss = (
            self.w_1 * l1_loss + 
            self.w_2 * l2_loss + 
            self.w_3 * sign_loss + 
            self.w_4 * grad_loss
        )

        return loss



class SignLoss(nn.Module):
    """
    BCE Loss for ensuring the sign of TSDF.
    """

    def __init__(self, reduction="mean"):
        super().__init__()
        self.bce = nn.BCELoss(reduction=reduction)

    def forward(self, input, target):
        bceInput = input / 2 + 0.5
        bceTarget = torch.sign(target)
        bceTarget[bceTarget < 0] = 0
        loss = self.bce(bceInput, bceTarget)
        return loss



class GradLoss(nn.Module):
    """
    Loss for voxel gradient.
    L1 error based on the 3-dimension sobel gradient.
    """

    def __init__(self, reduction="mean"):
        super().__init__()
        self.reduction = reduction

        self.sobel_x = torch.nn.Conv3d(1, 1, kernel_size=3, padding=1, bias=False)
        self.sobel_x.weight = torch.nn.Parameter(torch.from_numpy(np.array([
            [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
            [[-2, 0, 2], [-4, 0, 4], [-2, 0, 2]],
            [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
        ]) / 26. ).float().unsqueeze(0).unsqueeze(0))

        self.sobel_y = torch.nn.Conv3d(1, 1, kernel_size=3, padding=1, bias=False)
        self.sobel_y.weight = torch.nn.Parameter(torch.from_numpy(np.array([
            [[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
            [[-2, -4, -2], [0, 0, 0], [2, 4, 2]],
            [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]
        ]) / 26. ).float().unsqueeze(0).unsqueeze(0))

        self.sobel_z = torch.nn.Conv3d(1, 1, kernel_size=3, padding=1, bias=False)
        self.sobel_z.weight = torch.nn.Parameter(torch.from_numpy(np.array([
            [[-1, -2, -1], [-2, -4, -2], [-1, -2, -1]],
            [[ 0,  0,  0], [ 0,  0,  0], [ 0,  0,  0]],
            [[ 1,  2,  1], [ 2,  4,  2], [ 1,  2,  1]]
        ]) / 26. ).float().unsqueeze(0).unsqueeze(0))


    def forward(self, mask, input, target):
        inputGrad_x = self.sobel_x(input)
        inputGrad_y = self.sobel_y(input)
        inputGrad_z = self.sobel_z(input)

        targetGrad_x = self.sobel_x(target)
        targetGrad_y = self.sobel_y(target)
        targetGrad_z = self.sobel_z(target)

        loss_x = torch.abs(inputGrad_x[mask] - targetGrad_x[mask])
        loss_y = torch.abs(inputGrad_y[mask] - targetGrad_y[mask])
        loss_z = torch.abs(inputGrad_z[mask] - targetGrad_z[mask])

        if self.reduction == "mean":
            loss_x = loss_x.mean()
            loss_y = loss_y.mean()
            loss_z = loss_z.mean()
        elif self.reduction == "sum":
            loss_x = loss_x.sum()
            loss_y = loss_y.sum()
            loss_z = loss_z.sum()

        loss = (loss_x + loss_y + loss_z) / 3.
        return loss


class IoU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, target):
        inputRegion = (torch.abs(input) < 1)
        targetRegion = (torch.abs(target) < 1)
        intersection = torch.logical_and(inputRegion, targetRegion).sum()
        union = torch.logical_or(inputRegion, targetRegion).sum()
        return intersection / union
