# SSD300-in-face-detection-with-pytorch
Using SSD300 for face detection problem: Check is there any face in the image, if label is '1' (yes) => draw bounding boxes

- Test model with image in run.py
- Test model in realtime in Run_in_Realtime.py
<p></p>
- Model is trained by Wider_Face, how to get data is present in load_model.py
- Data preprocessing is deployed in data_preprcessing.py
- Create model in model_pretrained.py without loading weights

Hardware: Cuda - Nvidia GeForce GTX 3050Ti - Linux Ubuntu
Training: 100 epochs, batch_size=8
Framwork: Pytorch
