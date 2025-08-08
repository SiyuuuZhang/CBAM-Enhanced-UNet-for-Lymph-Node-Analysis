import glob
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset


class MyDataset(Dataset):
    def __init__(self, image_dir, label_dir):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.image_paths = sorted(glob.glob(str(image_dir / '*.png')))
        self.label_paths = sorted(glob.glob(str(label_dir / '*.png')))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image = Image.open(self.image_paths[index]).convert("RGB")
        label = Image.open(self.label_paths[index]).convert("L")
        compose = transforms.Compose([
            transforms.Resize((137, 137)),
            transforms.ToTensor()
        ])
        image = compose(image)
        label = compose(label)
        return image, label