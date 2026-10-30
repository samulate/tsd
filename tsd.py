import argparse
from pathlib import Path

import cv2
import numpy as np

if __name__ == "__main__": 
    parser = argparse.ArgumentParser(description='Detect traffic signs using a webcam.')
    parser.add_argument('--camera', type=int, default=0, help='Camera index')
    parser.add_argument('--device', default='cpu', help='cpu, mps, or a CUDA device such as 0')
    args = parser.parse_args()

    weights = Path(__file__).resolve().parent / 'models' / 'best.pt'
    if not weights.is_file():
        parser.error(f'Model weights not found: {weights}')

    from ultralytics import YOLO

    model = YOLO(str(weights))

    #Setting the resolution to transform the webcam stream to
    STREAMWIDTH = 1024
    STREAMHEIGHT = 576
    MINIMUM_CONFIDENCE = 0.4   
    RED = (0, 0, 255)
    CLASSNAMES = [
        'Added Lane',
        'Advisory Speed',
        'Dip',
        'Do Not Enter',
        'Do Not Pass',
        'Lane Ends Ahead',
        'Left Curve',
        'Merging Traffic',
        'Minimum Speed Limit', 
        'No Left Turn',
        'No Pedestrian Crossing',
        'No Right Turn',
        'No U-Turn',
        'One Direction Arrow',
        'One Way',
        'Pedestrian Crossing',
        'Right Curve',
        'Road Work Ahead',
        'Roundabout',
        'Sharp Left Turn',
        'Sharp Right Turn',
        'Speed Limit',
        'Stop Sign',
        'Traffic Signal Ahead',
        'Turn Left Only',
        'Turn Right Only',
        'Two Direction Arrow',
        'Yield'
    ]
    CLASSNAMES2 = [
        'Added Lane',
        'Advisory Speed',
        'Dip',
        'Do Not Enter',
        'Do Not Pass',
        'Lane Ends Ahead',
        'Left Curve',
        'Merging Traffic',
        'Minimum Speed Limit', 
        'No Left Turn',
        'No Pedestrian Crossing',
        'No Right Turn',
        'No U-Turn',
        'One Direction Arrow',
        'One Way',
        'Pedestrian Crossing',
        'Right Curve',
        'Road Work Ahead',
        'Roundabout',
        'Sharp Left Turn',
        'Sharp Right Turn',
        'Speed Limit',
        'Stop Sign',
        'Traffic Signal Ahead',
        'Turn Left Only',
        'Turn Right Only',
        'Two Direction Arrow',
        'Yield'
    ]
    #Get stream from first camera
    webcamStream = cv2.VideoCapture(args.camera)

    #Testing if I can open the webcam
    if not webcamStream.isOpened():
        webcamStream.release()
        parser.exit(1, f'Error: Could not open camera {args.camera}. Check camera permissions or try --camera 1.\n')
    
    #Changing the resolution
    webcamStream.set(cv2.CAP_PROP_FRAME_WIDTH, STREAMWIDTH)
    webcamStream.set(cv2.CAP_PROP_FRAME_HEIGHT, STREAMHEIGHT)

    try:
        while True:
            #Capture frame by frame
            ret, frame = webcamStream.read()

            if not ret:
                print("Error: Couldn't read a frame.")
                break

            frame = cv2.resize(frame, (1024, 576))

            #Filter the color out (make it black and white)
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            rgb_gray_frame = np.stack([gray_frame] * 3, axis=-1)

            #Use YOLO to detect on the frame
            detections = model(rgb_gray_frame, device=args.device, verbose=False)[0]

            #Loop over detections
            for data in detections.boxes.data.tolist():
                #Get confidence
                confidence = data[4]

                if float(confidence) < MINIMUM_CONFIDENCE:
                    continue

                xmin, ymin, xmax, ymax = int(data[0]), int(data[1]), int(data[2]), int(data[3])
                classid = data[5]
                classname = CLASSNAMES[int(classid)]
                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), RED, 2)
                cv2.putText(frame, str(classname), (xmin, ymin-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                cv2.putText(frame, str(round(confidence, 3)), (xmin, ymin-40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            #Display the frame
            cv2.imshow('Traffic Sign Detection', frame)

            #Exit loop with the 'q' key
            if cv2.waitKey(1) & 0xFF in (ord('q'), ord('`'), ord('~')):
                break
    finally:
        webcamStream.release()
        cv2.destroyAllWindows()



    
