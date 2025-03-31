import torch
import torchvision.transforms as transforms
from torchvision.datasets import WIDERFace
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import json

# Khởi tạo dataset
train_dataset = WIDERFace(
    root="./widerface",
    split="train",
    transform=transforms.ToTensor(),
    download=True
)

train_data_list = []

# Duyệt toàn bộ dataset
for i in range(len(train_dataset)):
    image, target = train_dataset[i]
    image = image.permute(1, 2, 0)
    bboxes = target["bbox"]
    image_path = train_dataset.img_info[i]["img_path"]
    label = 1 if len(bboxes) != 0 else 0

    train_data = {
        "image": image_path,
        "bboxes": bboxes.tolist(),  # Chuyển Tensor thành list để lưu JSON
        "labels": label
    }

    train_data_list.append(train_data)

# Lưu vào file JSON
with open("train_data.json", "w") as f:
    json.dump(train_data_list, f)

print(f"Lưu {len(train_data_list)} sample vào train_data.json")

# Khởi tạo dataset validation
val_dataset = WIDERFace(
    root="./widerface",
    split="val",
    transform=transforms.ToTensor(),
    download=True
)

val_data_list = []

# Duyệt toàn bộ dataset validation
for i in range(len(val_dataset)):
    image, target = val_dataset[i]
    bboxes = target["bbox"]
    image_path = val_dataset.img_info[i]["img_path"]
    label = 1 if len(bboxes) != 0 else 0

    val_data = {
        "image": image_path,
        "bboxes": bboxes.tolist(),  # Chuyển Tensor thành list để lưu JSON
        "labels": label
    }

    val_data_list.append(val_data)

# Lưu vào file JSON
with open("val_data.json", "w") as f:
    json.dump(val_data_list, f)

print(f"Lưu {len(val_data_list)} sample vào val_data.json")
