from ultralytics import YOLO

# Load a model
model = YOLO('./yolov8n.pt')  # load a pretrained model (recommended for training)
#../runs/detect/train/weights/last.pt to resume

# Train the model with 2 GPUs
results = model.train(data='./data.yaml', epochs=100, imgsz=640)
#results = model.train(resume=True, device=[0, 1])