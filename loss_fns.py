import torch

dd = 0.8
def l1(disp, _, gt, weight=None, trange=None):
    """
    Compute the L1-Loss

    :param disp: the disp output of the network
    :type disp: torch.Tensor

    :param gt: ground truth value of the pixel
    :type gt: torch.Tensor

    :param weight: per pixel weighting of loss
    :type weight: torch.Tensor

    :param trange: range to compute the loss
    :type trange: float

    :returns: the loss as torch.Tensor
    """
    loss = torch.abs(disp - gt)

    # ignore loss for out-of-range disparities
    if trange is not None:
        loss *= (torch.abs(gt) <= trange).float()
    
    if weight is not None:
        loss *= weight

    loss = torch.mean(loss)

    return loss


def mse(disp, _, gt, weight=None, trange=None):
    """
    Compute the Mean Squared Error

    :param disp: the disp output of the network
    :type disp: torch.Tensor

    :param gt: ground truth value of the pixel
    :type gt: torch.Tensor

    :param trange: range to compute the loss
    :type trange: float

    :param weight: per pixel weighting of loss
    :type weight: torch.Tensor

    :returns: the loss as torch.Tensor
    """
    loss = torch.pow(disp - gt, 2.0)

    # ignore loss for out-of-range disparities
    if trange is not None:
        loss *= (torch.abs(gt) <= trange).float()

    if weight is not None:
        loss *= weight

    loss = torch.mean(loss)

    return loss

def mse_uncert(disp, uncert, gt, weight=None, trange=None):
    """
    Compute the MSE-Loss for the disparity with uncertainty

    :param disp: the disp output of the network
    :type disp: torch.Tensor

    :param uncert: the uncertainty output of the network
    :type uncert: torch.Tensor

    :param gt: ground truth disparity
    :type gt: torch.Tensor

    :param weight: per pixel weighting of loss
    :type weight: torch.Tensor

    :param trange: range to compute the loss
    :type trange: float

    :returns: the loss as torch.Tensor
    """
    # compute loss with uncertainty
    loss = 0.5 * torch.exp(-uncert) * torch.pow(disp - gt, 2.0)

    # add uncertainty
    loss += 0.5 * uncert
    
    # ignore loss for out-of-range disparities
    if trange is not None:
        loss *= (torch.abs(gt) <= trange).float()

    if weight is not None:
        loss *= weight

    loss = torch.mean(loss)

    return loss

def l1_uncert(disp, uncert, gt, weight=None, trange=None):
    """
    Compute the L1-Loss for the disparity with uncertainty

    :param disp: the disp output of the network
    :type disp: torch.Tensor

    :param uncert: the uncertainty output of the network
    :type uncert: torch.Tensor

    :param gt: ground truth disparity
    :type gt: torch.Tensor

    :param weight: per pixel weighting of loss
    :type weight: torch.Tensor

    :param trange: range to compute the loss
    :type trange: float

    :returns: the loss as torch.Tensor
    """
    # compute loss with uncertainty
    loss = 0.5 * torch.exp(-uncert) * torch.abs(disp - gt)

    # add uncertainty
    loss += 0.5 * uncert
    
    # ignore loss for out-of-range disparities
    if trange is not None:
        loss *= (torch.abs(gt) <= trange).float()

    if weight is not None:
        loss *= weight

    loss = torch.mean(loss)

    return loss
