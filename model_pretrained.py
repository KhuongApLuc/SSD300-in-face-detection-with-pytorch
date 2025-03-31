import torchvision
import torch.optim as optim
from tqdm import tqdm
import torch
import cv2
import os
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from torchvision.models.detection.ssd import SSDClassificationHead
from torchvision.models.detection import _utils
from torchvision.models.detection import SSD300_VGG16_Weights

def create_model(num_classes=2, size=300):
    model = torchvision.models.detection.ssd300_vgg16(weights=SSD300_VGG16_Weights.COCO_V1)

    in_channels = _utils.retrieve_out_channels(model.backbone, (size, size))
    
    num_anchors = model.anchor_generator.num_anchors_per_location()
   
    model.head.classification_head = SSDClassificationHead(
        in_channels=in_channels,
        num_anchors=num_anchors,
        num_classes=num_classes
    )
    
    model.transform.min_size = (size,)
    model.transform.max_size = size

    return model

model = create_model(num_classes=2, size=640)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)