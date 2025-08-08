import os
import cv2
import glob
import torch
import numpy as np
import torchvision.transforms as transforms
from PIL import Image
from model import UNet
from pathlib import Path
from utils import DiceLoss, SensitivityCalculator, SpecificityCalculator


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UNet(n_channels=3, n_classes=1).to(device)
    DICE_criterion = DiceLoss()
    state_dict = torch.load('./checkpoints/model.pt')
    model.load_state_dict(state_dict)
    model.eval()
    image_compose = transforms.Compose([
        transforms.Resize((137, 137)),
        transforms.ToTensor(),
    ])
    image_dir = Path(r'./data/eval/img/')
    label_dir = Path(r'./data/eval/label/')
    image_paths = sorted(glob.glob(str(image_dir / '*.png')))
    label_paths = sorted(glob.glob(str(label_dir / '*.png')))
    for i in range(len(image_paths)):
        image = Image.open(image_paths[i]).convert("RGB")
        label = Image.open(label_paths[i]).convert("L")
        image = image_compose(image)
        image = torch.unsqueeze(image, 0).to(device)
        label = image_compose(label)
        label = torch.unsqueeze(label, 0)
        output = model(image)
        output = (output >= 0.5).int()
        output = output.squeeze().cpu()
        dice = DICE_criterion(output, label)
        sen = SensitivityCalculator(output, label)
        pre = SpecificityCalculator(output, label)
        output = output.detach().numpy()
        output = (output * 255).astype(np.uint8)
        filename = r'./output/' + os.path.basename(image_paths[i])
        print(filename, dice, sen, pre)
        cv2.imwrite(filename, output)


if __name__ == "__main__":
    main()