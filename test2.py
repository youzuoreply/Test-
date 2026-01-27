from ultralytics import YOLO

# Load a pretrained YOLOv8 model
model = YOLO('yolov8n.pt')  # nano model (fastest)

# Run inference on an image
results = model.predict(source='testimage.jpg')

# Display results
for result in results:
    print(result)
    result.save(filename='output.jpg')