from ultralytics import YOLO
model = YOLO("yolov8n.pt")
results = model.predict(source=r'D:\code\yolov8-custom-training\mouse.mp4',conf=0.8, show=True,save=True) # source already setup
names = model.names
# for r in results:
#     for c in r.boxes.cls:
#         print(names[int(c)])