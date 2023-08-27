from ultralytics import YOLO
import cv2
import numpy as np

model = YOLO('/Users/samvathk/Desktop/Traffic Sign Detection 2/runs/detect/epoch500tsd/train/weights/best.pt')
if __name__ == "__main__": 

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
    webcamStream = cv2.VideoCapture(0)

    #Testing if I can open the webcam
    if not webcamStream.isOpened():
        print("Error: Could not open camera.")
        exit()
    
    #Changing the resolution
    webcamStream.set(cv2.CAP_PROP_FRAME_WIDTH, STREAMWIDTH)
    webcamStream.set(cv2.CAP_PROP_FRAME_HEIGHT, STREAMHEIGHT)

    while True:
        #Capture frame by frame
        ret, frame = webcamStream.read()

        if not ret:
            print("Error: Couldn't read a frame.")
            continue

        frame = cv2.resize(frame, (1024, 576))


        #If the frame is retrieved succesfully
        if ret: 

            #Filter the color out (make it black and white)
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            #gray_frame = gray_frame / 255.0
            rgb_gray_frame = np.stack([gray_frame] * 3, axis=-1)


            gray_frame = np.expand_dims(gray_frame, axis=0)
            gray_frame = np.expand_dims(gray_frame, axis=-1)
            print(rgb_gray_frame.shape)

            #Use YOLO to detect on the frame
            detections = model(rgb_gray_frame)[0]

            #Initialize Results
            results = []

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
            if cv2.waitKey(1) & 0xFF == ord('`') or cv2.waitKey(1) & 0xFF == ord('~'):
                break
        
        else:
            print("Error: Could not read frame.")
            break
    
    #Release the capture and close all the windows when done
    webcamStream.release()
    cv2.destroyAllWindows()



    