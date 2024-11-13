import cv2
from ultralytics import YOLO


# Initialize the YOLO model
class YoLo_detect:
    """
    A class for performing object detection using YOLOv8.

    Attributes:
        model (YOLO): The YOLO model loaded from the specified model path.
    """

    def __init__(self, model_path):
        """
        Initializes the YoLo_detect class with a YOLO model.

        Args:
            model_path (str): Path to the YOLO model file.
        """
        self.model = YOLO(model_path)
        
    def detect_image(self, image_path: str):
        """
        Detects objects in an image and saves the annotated result.

        Args:
            image_path (str): Path to the input image.
            output_path (str): Path to save the annotated image. Default is "output.jpg".

        Returns:
            None
        """
        # read input image file
        img = cv2.imread(image_path)
        if img is None:
            print(f"Failed to load image from {image_path}")
            return
        
        # inference yolo to determine
        results = self.model(img)
        annotated_image = results[0].plot()  # Annotate hình ảnh

        # show results
        cv2.imshow("Detection Results", annotated_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def read_cap(self,cap):
        """   Reads frames from the webcam or video capture object, performs inference using the YOLO model,
        annotates the frames with detected objects, and displays them in a window. The loop continues 
        until the user presses 'q' to exit or the video stream ends.

        Args:
            cap (cv2.VideoCapture): The OpenCV capture object used to read frames from a webcam or video file.
                                This is usually initialized with cv2.VideoCapture() with the camera index or file path.
        """
        if not cap.isOpened():
            print(f"Cannot open capture")
            return
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Finished processing the video or unable to read the frame.")
                break

            #  Perform inference on the frame
            results = self.model(frame)

            # Annotate the frame with bounding boxes and labels
            annotated_frame = results[0].plot()  # Automatically draws bounding boxes and labels on the frame

            # Display the annotated frame
            cv2.imshow("YOLOv8 Inference", annotated_frame)



            # Press 'q' to exit the video processing
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # Release resources
        cap.release()
        cv2.destroyAllWindows()
    def detect_video(self, video_path: str):
        """
        Detects objects in a video and saves the annotated results.

        Args:
            video_path (str): Path to the input video.
            output_path (str): Path to save the annotated video. Default is "output_results/annotated_video.mp4".

        Returns:
            None
        """
        # Read the input video
        cap = cv2.VideoCapture(video_path)
        self.read_cap(cap)
        
    def detect_webcam(self):
        """
        Detects objects in the webcam and saves the annotated results.

        Returns:
            None
        """
        cap = cv2.VideoCapture(0)
        self.read_cap(cap)
    

