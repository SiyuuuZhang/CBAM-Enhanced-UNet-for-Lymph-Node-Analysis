import torch
import torch.nn as nn


class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-5):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, predicted, target):
        predicted = predicted.view(-1)
        target = target.view(-1)
        flipped_predicted = 1 - predicted
        flipped_target = 1 - target
        intersection = torch.sum(flipped_predicted * flipped_target)
        union = torch.sum(flipped_predicted) + torch.sum(flipped_target)
        dice = (2. * intersection) / (union + self.smooth)
        dice_loss = 1 - dice
        return dice_loss

def SensitivityCalculator(predictions, labels):
    binary_predictions = (predictions >= 0.5).int()
    true_positives = torch.sum((binary_predictions == 0) & (labels == 0))
    sensitivity = (true_positives * 1. / torch.sum(labels == 0))
    return sensitivity

def SpecificityCalculator(predictions, labels):
    binary_predictions = (predictions >= 0.5).int()
    false_positives = torch.sum((binary_predictions == 1) & (labels == 1))
    specificity = (false_positives * 1. / torch.sum(labels == 1))
    return specificity