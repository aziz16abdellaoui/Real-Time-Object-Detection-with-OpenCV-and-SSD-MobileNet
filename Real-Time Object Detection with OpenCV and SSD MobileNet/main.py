import cv2

# Function to detect objects in an image file
def ImgFile():
    img = cv2.imread('person.png')

    # Load class names from file
    classNames = []
    classFile = 'coco.names'
    with open(classFile, 'rt') as f:
        classNames = f.read().rstrip('\n').split('\n')

    # Load model configuration and weights
    configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
    weightpath = 'frozen_inference_graph.pb'

    net = cv2.dnn_DetectionModel(weightpath, configPath)
    net.setInputSize(320, 230)
    net.setInputScale(1.0 / 127.5)
    net.setInputMean((127.5, 127.5, 127.5))
    net.setInputSwapRB(True)

    # Detect objects in the image
    classIds, confs, bbox = net.detect(img, confThreshold=0.5)

    for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
        cv2.rectangle(img, box, color=(0, 255, 0), thickness=2)
        cv2.putText(img, classNames[classId - 1], (box[0] + 10, box[1] + 20), 
                    cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), thickness=2)

    # Display the processed image
    cv2.imshow('Processed Image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Function to detect objects using a webcam
def Camera():
    cam = cv2.VideoCapture(0)  # Use the default PC camera
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # Load class names from file
    classNames = []
    classFile = 'coco.names'
    with open(classFile, 'rt') as f:
        classNames = f.read().rstrip('\n').split('\n')

    # Load model configuration and weights
    configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
    weightpath = 'frozen_inference_graph.pb'

    net = cv2.dnn_DetectionModel(weightpath, configPath)
    net.setInputSize(320, 320)
    net.setInputScale(1.0 / 127.5)
    net.setInputMean((127.5, 127.5, 127.5))
    net.setInputSwapRB(True)

    while True:
        success, img = cam.read()
        if not success:
            print("Failed to read from camera")
            break

        # Detect objects in the webcam image
        classIds, confs, bbox = net.detect(img, confThreshold=0.6)

        if len(classIds) != 0:
            for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
                cv2.rectangle(img, box, color=(0, 255, 0), thickness=2)
                cv2.putText(img, classNames[classId - 1], (box[0] + 10, box[1] + 20), 
                            cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), thickness=2)

        # Display the processed video feed
        cv2.imshow('Processed Video', img)

        # Exit on pressing the 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cam.release()
    cv2.destroyAllWindows()

# Uncomment one of the below lines to use the respective function
# ImgFile()
Camera()
