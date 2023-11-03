from ultralytics import YOLO
model = YOLO("data_TD_items_5_cls.pt")
results = model.predict(source=r'D:\code\yolov8\testing',conf=0.5, show=True,save=True) # source already setup
names = model.names
for r in results:
    for c in r.boxes.cls:
        print(names[int(c)])