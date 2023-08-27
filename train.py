from ultralytics import YOLO

#load model
model = YOLO('yolov8n.pt')


#use model
if __name__ == '__main__':
    results = model.train(data="C:\\Users\\samva\\YOLO Datasets\\trafficsigndetection.v4i.yolov8\\data.yaml", epochs=500, device=0, imgsz=1024, patience=0)
    results = model.val()