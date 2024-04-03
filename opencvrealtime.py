from ultralytics import YOLO
model = YOLO("best3.pt")
result = model.predict(source="0", show=True)
print(result)