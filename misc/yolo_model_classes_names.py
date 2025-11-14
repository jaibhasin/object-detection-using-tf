from ultralytics import YOLO
model = YOLO("models/research/object_detection/yolov8s_oiv7/yolov8s-oiv7.pt")
print(model.names)
