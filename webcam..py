import cv2
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFilter
# Load the YOLOv8 model
model = YOLO(r'C:\Projects\gdsc_lemon\best (1).pt')

def merge_roi(frame, roi, x, y):
    frame[y:y+200, x:x+200, :] = roi
    return frame

def create_roi(frame,upperleft,bottomright) :
    r=cv2.rectangle(frame,upperleft,bottomright,(200, 100, 200), 5)
    rect_img=frame[upperleft[1] : bottomright[1], upperleft[0] : bottomright[0]]
    return rect_img
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW) # this is the magic!

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1600)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1200)
upper_left = (500, 200)
bottom_right = (1500, 1200)
# Loop through the video frames
while cap.isOpened():
    # Read a frame from the videoq
    success, frame = cap.read()

    if success:
        # Run YOLOv8 inference on the frame
        # Crop a region of interest (ROI) from the frame

        # Resize the ROI to a specific size (e.g., 200x200)
        # roi_resized = cv2.resize(roi, (200, 200))

        # Merge the resized ROI back into the frame
        roi=create_roi(frame,upper_left,bottom_right)
        results = model(roi)

        # frame = merge_roi(frame, roi_resized, 0, 0)

        #
        # # Visualize the results on the frame
        annotated_frame = results[0].plot()
        frame[upper_left[1]: bottom_right[1], upper_left[0]: bottom_right[0]] = annotated_frame

        # Display the annotated frame
        cv2.imshow("YOLOv8 Inference", frame)
        if cv2.waitKey(1) == ord('q'):

            break
cap.release()
cv2.destroyAllWindows()