import json
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os

class WiderFaceDataset(Dataset):
    def __init__(self, json_file, image_root, transform=None):
        with open(json_file, "r") as f:
            self.data = json.load(f)
        
        self.image_root = image_root
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        image_path = os.path.join(self.image_root, sample["image"])
        image = Image.open(image_path).convert("RGB")

        bboxes = []
        for x, y, w, h in sample["bboxes"]:
            x_max, y_max = x + w, y + h
            if w > 0 and h > 0:  # Chỉ giữ bbox hợp lệ
                bboxes.append([x, y, x_max, y_max])

        if len(bboxes) == 0:
            return self.__getitem__((idx + 1) % len(self.data))

        bboxes = torch.tensor(bboxes, dtype=torch.float32)
        labels = torch.ones((bboxes.shape[0],), dtype=torch.int64)

        target = {"boxes": bboxes, "labels": labels}

        if self.transform:
            image = self.transform(image)

        return image, target

transform = transforms.Compose([
    #transforms.Resize((300, 300)),
    transforms.ToTensor(),
])

train_dataset = WiderFaceDataset(json_file="train_data.json", image_root="./widerface", transform=transform)
train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True, collate_fn=lambda batch: tuple(zip(*batch)))

val_dataset = WiderFaceDataset(json_file="val_data.json", image_root="./widerface", transform=transform)
val_dataloader = DataLoader(val_dataset, batch_size=8, shuffle=True, collate_fn=lambda batch: tuple(zip(*batch)))