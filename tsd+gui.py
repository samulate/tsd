from ultralytics import YOLO
from pathlib import Path
import cv2
import numpy as np
import tkinter as tk
from tkinter import PanedWindow, Frame
from PIL import Image, ImageTk

if __name__ == "__main__":
    weights = Path(__file__).resolve().parent / 'models' / 'best.pt'
    if not weights.is_file():
        raise SystemExit(f'Model weights not found: {weights}')
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
    window = tk.Tk()
    window.title("Traffic Sign Detection+")
    images = []

    #Get stream from first camera
    webcamStream = cv2.VideoCapture(0)

    def closeWindow():
        webcamStream.release()
        window.destroy()

    window.protocol('WM_DELETE_WINDOW', closeWindow)

    #Testing if I can open the webcam
    if not webcamStream.isOpened():
        closeWindow()
        raise SystemExit('Error: Could not open camera. Check camera permissions.')
    
    #Changing the resolution
    webcamStream.set(cv2.CAP_PROP_FRAME_WIDTH, STREAMWIDTH)
    webcamStream.set(cv2.CAP_PROP_FRAME_HEIGHT, STREAMHEIGHT)

    #Resize event
    def resizeEvent(event):
        global canvasWidth
        canvasWidth = event.width
        panedWindow.sash_place(0, canvasWidth, window.winfo_height() - 100)

    def updateVideo():
        global canvasWidth

        #Capture frame by frame
        ret, frame = webcamStream.read()

        if not ret:
            print("Error: Couldn't read a frame.")
            closeWindow()
            return

        frame = cv2.resize(frame, (1024, 576))


        #If the frame is retrieved succesfully
        if ret: 

            #Filter the color out (make it black and white)
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            #gray_frame = gray_frame / 255.0
            rgb_gray_frame = np.stack([gray_frame] * 3, axis=-1)


            gray_frame = np.expand_dims(gray_frame, axis=0)
            gray_frame = np.expand_dims(gray_frame, axis=-1)

            #Use YOLO to detect on the frame
            detections = model(rgb_gray_frame, device='cpu', verbose=False)[0]

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

            #Convert the OpenCV frame to a format tkinter can use and display it
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (canvasWidth, int(canvasWidth * (576/1024)))) #Maintain aspect ratio
            photo = ImageTk.PhotoImage(image=Image.fromarray(frame))
            canvas.create_image(0, 0, image=photo, anchor=tk.NW)

            #Add a photo to the list and remove old images
            images.append(photo)
            if len(images) > 1:
                images.pop(0)

            window.after(30, updateVideo)

    canvasWidth = 1024

    
    panedWindow = PanedWindow(window, orient=tk.VERTICAL)
    panedWindow.pack(fill=tk.BOTH, expand=1)

    #Create a canvas to display the video stream
    canvas = tk.Canvas(panedWindow, width=canvasWidth, height=int(canvasWidth * (576/1024)))
    panedWindow.add(canvas)

    #Create a Frame for buttons and add to PanelWindow
    buttonFrameHeight = 80
    buttonFrame = Frame(panedWindow, height=buttonFrameHeight)
    panedWindow.add(buttonFrame)

    #Add a sample button
    sampleButton = tk.Button(buttonFrame, text="Sample Button", command=lambda: print("Button Pressed"))
    sampleButton.pack(pady=20)

    window.bind("<Configure>", resizeEvent)

    #Start updating the video display
    try:
        updateVideo()

        #Run the tkinter main loop
        window.mainloop()
    finally:
        webcamStream.release()
