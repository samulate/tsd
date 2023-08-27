from ultralytics import YOLO

#load model
model = YOLO('/Users/samvathk/Desktop/Traffic Sign Detection 2/runs/detect/epoch500tsd/train/weights/best.pt')

if __name__ == '__main__':
    model.predict("/Users/samvathk/Desktop/YOLO/datasets/trafficsigndetection.v4i.yolov8/valid/images/addedLane_1331866567-avi_image8_png.rf.b757769620b113fd9da3444f503badee.jpg", save=True, imgsz=1024, device='mps', show=True)