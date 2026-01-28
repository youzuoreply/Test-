from ultralytics import YOLO

print("i changed")
model = YOLO('yolov8n.pt')  # nano model (fastest)


results = model.predict(source='testimage.jpg')

for result in results:
    print(result)
    result.sa