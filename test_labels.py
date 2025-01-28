from roboflow import Roboflow
from ultralytics import YOLO

rf = Roboflow(api_key="0gmqsn7ulWkoDTh7TMog")
project = rf.workspace("parking-szukg").project("license_plate_detection-cnpxn")
version = project.version(2)
dataset = version.download("yolov8")

# Create YOLO model
model = YOLO('yolov8n.pt')  # Start with pre-trained nano model

# Train the model
results = model.train(
    data=f'{dataset.location}/data.yaml',  # Path to dataset configuration
    epochs=100,  # Number of training epochs
    imgsz=640,  # Input image size
    batch=16,   # Batch size
    device='cpu'    # GPU device (change to 'cpu' if no GPU)
)

# Validate the trained model
metrics = model.val()

# Save the best model
model.save('labels.pt')