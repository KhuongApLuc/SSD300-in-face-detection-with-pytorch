import torchvision
import torch.optim as optim
from tqdm import tqdm
import torch
from torchvision.models.detection.ssd import SSDClassificationHead
from torchvision.models.detection import _utils
from torchvision.models.detection import SSD300_VGG16_Weights
from data_preprocessing import *
from model_pretrained import *

params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.0001, momentum=0.9)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
num_epochs = 100

for epoch in range(num_epochs):
    print(f"\n🔹 Epoch {epoch+1}/{num_epochs} - Training...")

    model.train()
    train_running_loss = 0.0
    progress_bar = tqdm(train_dataloader, desc=f"Train {epoch+1}", leave=False)

    for images, targets in progress_bar:
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        optimizer.zero_grad()
        loss_dict = model(images, targets)
        loss = sum(loss for loss in loss_dict.values())
        loss.backward()
        optimizer.step()

        train_running_loss += loss.item()
        progress_bar.set_postfix(loss=f"{loss.item():.4f}")

    epoch_train_loss = train_running_loss / len(train_dataloader)
    print(f"✅ Epoch {epoch+1} - Avg Train Loss: {epoch_train_loss:.4f}")

    print(f"\n🔹 Epoch {epoch+1}/{num_epochs} - Validation...")

torch.save(model, "model.pth")
torch.save(model.state_dict(), "model_std.pth")