import cv2
import os
from PIL import Image
import numpy as np

def overlay_image_on_camera_stream(image_path, alpha=0.5, camera_index=0):
    """
    Overlays a single image onto a camera stream in real-time.

    Parameters:
        image_path (str): Path to the image file to be overlayed.
        alpha (float): Transparency level of the overlay image (0.0 to 1.0).
        camera_index (int): Index of the camera for cv2.VideoCapture.
    """
    # Capture the video stream from the specified camera
    cap = cv2.VideoCapture(camera_index)
    
    # Load the overlay image
    overlay_image = Image.open(image_path).convert("RGBA")

    while cap.isOpened():
        # Read frame from the camera
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

        # Resize overlay image to match the frame size if necessary
        overlay_resized = overlay_image.resize((frame.shape[1], frame.shape[0]))

        # Convert the camera frame to a format compatible with PIL
        frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        # Blend the overlay image with the frame
        blended_frame = Image.blend(frame_pil, overlay_resized, alpha=alpha)

        # Convert back to OpenCV format and display
        blended_frame_cv2 = cv2.cvtColor(np.array(blended_frame), cv2.COLOR_RGB2BGR)
        cv2.imshow("Overlayed Camera Stream", blended_frame_cv2)

        # Exit on pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
