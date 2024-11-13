from model import yolo_detector 

# Define the path model to use
YOLO_PATH = '13_11_inference_yolo/model/yolov8n.pt'


yolo = yolo_detector.YoLo_detect(YOLO_PATH)

# Detecting an image
# yolo.detect_image('13_11_inference_yolo/data/input_data/images.jpeg')


# Detecting a video
# yolo.detect_video('13_11_inference_yolo/input_data/project_video.mp4')


# Detecting webcam
yolo.detect_webcam() 