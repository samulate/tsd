import cv2
import tkinter as tk
from PIL import Image, ImageTk
from tkinter import PanedWindow, Frame

images = []

#Resize event
def resizeEvent(event):
    global canvasWidth
    canvasWidth = event.width
    panedWindow.sash_place(0, canvasWidth, window.winfo_height() - 100)

# This function is triggered when the button is pressed
def toggle_filter():
    global apply_filter
    apply_filter = not apply_filter

# Set up the main tkinter window
window = tk.Tk()
window.title("OpenCV GUI with Filter Toggle")

# Variable to decide whether to apply a filter
apply_filter = False

# Capture video from default camera (you can change 0 to a different source if needed)
cap = cv2.VideoCapture(0)

# Display the video stream in the tkinter window and update it
def update_video():
    ret, frame = cap.read()

    if not ret:
        print('Error: Failed to grab frame')
        return

    # Apply a filter if apply_filter is True (In this case, we'll convert the frame to grayscale)
    if apply_filter:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)  # Convert grayscale back to RGB for tkinter

    # Convert the OpenCV frame to a format tkinter can use and display it
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = cv2.resize(frame, (canvasWidth, int(canvasWidth * (576/1024)))) #Maintain aspect ratio
    photo = ImageTk.PhotoImage(image=Image.fromarray(frame))
    canvas.create_image(0, 0, image=photo, anchor=tk.NW)

    # Add photo to the list and remove old images
    images.append(photo)
    if len(images) == 2: 
        images.pop(0)

    window.after(10, update_video)

canvasWidth = 1024

panedWindow = PanedWindow(window, orient=tk.VERTICAL)
panedWindow.pack(fill=tk.BOTH, expand=1)



# Create a canvas to display the video stream
canvas = tk.Canvas(window, width=canvasWidth, height=int(canvasWidth * (576/1024)))
canvas.pack()

# Add a button to toggle the filter on/off
btn_toggle = tk.Button(window, text="Toggle Filter", command=toggle_filter)
btn_toggle.pack()

# Start updating the video display
update_video()

# Run the tkinter main loop
window.mainloop()
