import cv2
import torch
from PIL import Image
import numpy as np
import torchvision

image_path = r"D:\Bac_cc\482012613_1395072671848675_3889318782743149531_n.jpg"
output_path = "output_check.jpg"
image = Image.open(image_path).convert("RGB")

image_np = np.array(image)

device = torch.device('cpu')
model = torchvision.models.detection.ssdlite320_mobilenet_v3_large(pretrained=False, num_classes=2)
model=torch.load("D:\Bac_cc\model.pth", map_location=device, weights_only=False)
model.to(device)
model.eval()

model.eval()
with torch.no_grad():
    predictions = model([torchvision.transforms.ToTensor()(image).to(device)])

boxes = predictions[0]['boxes'].cpu().numpy()
labels = predictions[0]['labels'].cpu().numpy()
scores = predictions[0]['scores'].cpu().numpy()

for box, label, score in zip(boxes, labels, scores):
    if score > 0.1:
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(image_np, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image_np, f"{label}: {score:.2f}", (x1, y1 - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

cv2.imwrite(output_path, cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))

print(f"✅ Ảnh đã được lưu vào {output_path}")
