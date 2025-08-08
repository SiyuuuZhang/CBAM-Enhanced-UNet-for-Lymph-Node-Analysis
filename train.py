import torch
import numpy as np
import torch.nn as nn
from model import UNet
from torch import optim
from pathlib import Path
from dataset import MyDataset
from torch.utils.data import DataLoader
from utils import DiceLoss, SensitivityCalculator, SpecificityCalculator


def main():
    train_dataset = MyDataset(Path("./data/train/img"), Path("./data/train/label"))
    eval_dataset = MyDataset(Path("./data/eval/img"), Path("./data/eval/label"))
    train_dataloader = DataLoader(train_dataset, batch_size=6, shuffle=True)
    eval_dataloader = DataLoader(eval_dataset, batch_size=1, shuffle=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UNet(n_channels=3, n_classes=1).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=0.0075, weight_decay=0.005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 10, gamma=0.95)
    DICE_criterion = DiceLoss()
    BCE_criterion = nn.BCELoss()
    best_dice = 1.0
    for epoch in range(100):
        total_loss = 0.0
        total_size = 0.0
        model.train()
        for image, label in train_dataloader:
            optimizer.zero_grad()
            image = image.to(device=device, dtype=torch.float32)
            label = label.to(device=device, dtype=torch.float32)
            pred = model(image)
            loss = DICE_criterion(pred, label) + BCE_criterion(pred, label)
            loss.backward()
            optimizer.step()
            total_loss += loss.cpu().item()
            total_size += 1
        lr_scheduler.step()
        print('epoch', epoch, 'Loss:', total_loss / total_size)
        Dice_log = []
        Sen_log = []
        Spe_log = []
        model.eval()
        with torch.no_grad():
            for image, label in eval_dataloader:
                image = image.to(device=device, dtype=torch.float32)
                label = label.to(device=device, dtype=torch.float32)
                pred = model(image)
                pred = pred.cpu()
                label = label.cpu()
                dice = DICE_criterion(pred, label)
                sen = SensitivityCalculator(pred, label)
                spe = SpecificityCalculator(pred, label)
                Dice_log.append(dice)
                Sen_log.append(sen)
                Spe_log.append(spe)
            print(
                "epoch %d evaluated data average Dice = %f, Sensitivity = %f, Specificity = %f" %
                (epoch, np.mean(Dice_log), np.mean(Sen_log), np.mean(Spe_log)))
        if np.mean(Dice_log) < best_dice:
            best_dice = np.mean(Dice_log)
            torch.save(model.state_dict(), './checkpoints/model%d.pt' % epoch)


if __name__ == "__main__":
    main()