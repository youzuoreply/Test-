from ultralytics import YOLO

print("i changed")
model = YOLO('yolov8n.pt')  # nano model (fastest)


results = model.predict(source='testimage.jpg')

for result in results:
    print(result)
    result.save(filename='output.jpg')

print("Inference complete. Results saved to output.jpg")

if True:
    print("This block always executes.")