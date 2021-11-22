import numpy
import torch
import torch.nn.functional as torch_func

from torch import nn
from ece9603_project.lovaszHinge import lovasz_hinge

"""
A collection of interesting and useful loss functions for image classification
and segmentation.  Many of these loss functions are discussed and compared in:
https://arxiv.org/pdf/2006.14822.pdf
"""

def sigmoidAndFlatten(predictions, targets):
    # Don't use this if model contains sigmoid style activation layer
    predictions = torch_func.sigmoid(predictions)

    # Flatten target and predicted values
    predictions = predictions.view(-1)
    targets = targets.view(-1)

    return predictions, targets


class DiceLoss(nn.Module):
    """
    The Dice coefficient is a good metric for determining the similarity between
    two images.  Recently it has been adapted for machine learning applications
    as a loss function

    Parameters
    smooth: Present to prevent the function from becoming an undefined value
            from being used in edge cases
    """
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, predictions, targets, smooth=1):
        predictions, targets = sigmoidAndFlatten(predictions, targets)

        intersection = (predictions * targets).sum()
        dice = (2.*intersection + smooth) / (predictions.sum() + targets.sum() + smooth)

        return 1-dice


class BCEDiceLoss(nn.Module):
    """
    A combination of the dice and binary cross-entropy (BCE) loss functions.
    This combination introduces the stability of the binary cross-entropy loss
    function to the useful dices loss function
    """
    def __init__(self, weight=None, size_average=True):
        super(BCEDiceLoss, self).__init__()

    def forward(self, predictions, targets, smooth=1):
        predictions, targets = sigmoidAndFlatten(predictions, targets)

        # Dice Loss Part
        intersection = (predictions * targets).sum()
        dice = (2.*intersection + smooth) / (predictions.sum() + targets.sum() + smooth)
        dice_loss = 1 - dice

        # BCE Loss Part
        bce_loss = torch_func.binary_cross_entropy(predictions, targets, reduction='mean')

        return dice_loss + bce_loss


class JaccardIoULoss(nn.Module):
    """

    """
    def __init__(self, weight=None, size_average=True):
        super(JaccardIoULoss, self).__init__()

    def forward(self, predictions, targets, smooth=1):
        predictions, targets = sigmoidAndFlatten(predictions, targets)

        # Intersection can be considered the same as True Positive count
        intersection = (predictions * targets).sum()
        # Union includes the entire set of predictions and targets combined
        total = (predictions + targets).sum()
        union = total - intersection

        IntersectionOverUnion = (intersection + smooth) / (union + smooth)

        return 1 - IntersectionOverUnion


class FocalLoss(nn.Module):
    """
    Publication: https://arxiv.org/abs/1708.02002

    Focal Loss can be considered a variation of binary cross-entropy.  It reduces
    the significance of contributions from "easier" examples which forces the
    model to learn more from "harder" examples making it ideal for imbalanced
    class scenarios

    Parameters
    alpha: Must be in the range [0,1]
    gamma: Must be >0, when gamma=1 Focal Loss will behave like binary cross-entropy
    """
    def __init__(self, weight=None, size_average=True):
        super(FocalLoss, self).__init__()

    def forward(self, predictions, targets, alpha=0.8, gamma=2, smooth=1):
        predictions, targets = sigmoidAndFlatten(predictions, targets)

        bce_loss = torch_func.binary_cross_entropy(predictions, targets, reduction='mean')
        bce_exp = torch.exp(-bce_loss)
        focal_loss = alpha * (1-bce_exp) ** gamma * bce_loss

        return focal_loss


class TverskyLoss(nn.Module):
    """
    Publication: https://arxiv.org/abs/1706.05721

    Tversky Loss can be considered a variation of the Dice coefficient with
    better generalization.  The alpha and beta coefficient helps add weight to
    false positives and false negatives respectively.

    Parameters
    alpha: Adjustable weight for false positives
    beta: Adjustable weight for false negatives
    """
    def __init__(self, weight=None, size_average=True):
        super(TverskyLoss, self).__init__()

    def forward(self, predictions, targets, smooth=1, alpha=0.5, beta=0.5):
        predictions, targets = sigmoidAndFlatten(predictions, targets)

        # Calculate amount of True Positives, False Positives and False Negatives
        TP = (predictions * targets).sum()
        FP = ((1-targets) * predictions).sum()
        FN = (targets * (1-predictions)).sum()

        tversky = (TP + smooth) / (TP + alpha*FP + beta*FN + smooth)

        return 1 - tversky

class FocalTverskyLoss(nn.Module):
    """
    A variation of the Tversky loss function which utilizes the gamma parameter
    from the Focal loss function.

    Parameters
    alpha: Adjustable weight for false positives
    beta: Adjustable weight for false negatives
    gamma: Focal loss modifier must be in range [1,3]
    """
    def __init__(self, weight=None, size_average=True):
        super(FocalTverskyLoss, self).__init__()

    def forward(self, predictions, targets, smooth=1, alpha=0.5, beta=0.5, gamma=1):
        predictions, targets = sigmoidAndFlatten(predictions, targets)

        # Calculate amount of True Positives, False Positives and False Negatives
        TP = (predictions * targets).sum()
        FP = ((1-targets) * predictions).sum()
        FN = (targets * (1-predictions)).sum()

        tversky = (TP + smooth) / (TP + alpha*FP + beta*FN + smooth)
        focalTversky = (1 - tversky) ** gamma

        return focalTversky

# TODO: Get LovaszHingeLoss working
class LovaszHingeLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(LovaszHingeLoss, self).__init__()

    def forward(self, predictions, targets):
        inputs = torch_func.sigmoid(predictions)

        return lovasz_hinge(predictions, targets, per_image=False)

lossFunctionMap = {
    'bce_with_logits_loss': nn.BCEWithLogitsLoss(),
    'dice_loss': DiceLoss(),
    'bce_dice_loss': BCEDiceLoss(),
    'jaccard_iou_loss': JaccardIoULoss(),
    'focal_loss': FocalLoss(),
    'tversky_loss': TverskyLoss(),
    'focal_tversky_loss': FocalTverskyLoss()
}

def getLossFunction(name):
    return lossFunctionMap[name]
