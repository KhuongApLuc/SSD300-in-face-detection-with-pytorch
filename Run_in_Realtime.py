import cv2
import torch
from PIL import Image
import torchvision.transforms as T

model = torch.load("D:\Bac_cc\model.pth", map_location=torch.device("cpu"), weights_only=False)
model.eval()

cap = cv2.VideoCapture(0)
threshold = 0.6

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    transform = T.Compose([T.ToPILImage(), T.ToTensor()])
    img_tensor = transform(frame).unsqueeze(0)

    with torch.no_grad():
        prediction = model(img_tensor)

    for i in range(len(prediction[0]['boxes'])):  # Duyệt theo chỉ số i
        score = prediction[0]['scores'][i].item()
        if score >= threshold:
            x1, y1, x2, y2 = map(int, prediction[0]['boxes'][i])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{score:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imshow("Real-time Object Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
