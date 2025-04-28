Code Explanation: Emotion & Activity Detection Using YOLO and Two CNN Models
This script performs real-time emotion and activity detection using two different CNN models. It detects faces using YOLOv3, extracts face regions, and classifies emotions (Boredom, Confusion, Engagement, Frustration) and activities (Sleep, Yawn, Active) separately.

Step-by-Step Breakdown:
1. Load the Required Libraries
python
Copy
Edit
import cv2
import numpy as np
import tensorflow as tf
import time
from collections import deque
cv2: For video capture and face detection using YOLO.

numpy: For array manipulations.

tensorflow: For loading and running deep learning models.

time: Used for controlling activity detection every 10 seconds.

deque (from collections): Used for maintaining a rolling history of emotion predictions to avoid flickering.

2. Load the Pretrained CNN Models
python
Copy
Edit
# ✅ Load the emotion detection model
emotion_model = tf.keras.models.load_model('my_model.h5')

# ✅ Load the activity detection model
activity_model = tf.keras.models.load_model('vgg_model_3_labels.h5')
emotion_model predicts one of the four emotions:
['Boredom', 'Confusion', 'Engagement', 'Frustration']

activity_model predicts one of the three activities:
['Sleep', 'Yawn', 'Active']

Each model is loaded from a saved .h5 file.

3. Initialize YOLO for Face Detection
python
Copy
Edit
yolo_net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
yolo_net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
yolo_net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
YOLO (You Only Look Once) is used for real-time face detection.

GPU Acceleration: The setPreferableBackend and setPreferableTarget functions enable CUDA (GPU processing) for faster detection.

python
Copy
Edit
layer_names = yolo_net.getLayerNames()
output_layers = [layer_names[i - 1] for i in yolo_net.getUnconnectedOutLayers()]
Retrieves the output layers for YOLO to process detections.

4. Define Labels & History Buffers
python
Copy
Edit
# Labels
emotion_labels = ['Boredom', 'Confusion', 'Engagement', 'Frustration']
activity_labels = ['Sleep', 'Yawn', 'Active']

# Rolling history for smoothing predictions
emotion_history = deque(maxlen=5)
last_activity_time = time.time()
emotion_labels and activity_labels store class names for predictions.

deque(maxlen=5): Stores the last 5 emotion predictions to avoid rapid fluctuations in detected emotions.

last_activity_time = time.time(): Keeps track of the last activity detection time to limit activity predictions to every 10 seconds.

5. Capture Video Stream
python
Copy
Edit
cap = cv2.VideoCapture("happy.mp4")  # Change this to 0 for webcam input
Captures video from a file (happy.mp4).

Use cap = cv2.VideoCapture(0) for real-time webcam input.

6. Process Each Video Frame
python
Copy
Edit
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
Loops through each frame in the video.

If the frame is invalid (ret == False), the loop stops.

7. Prepare the Frame for YOLO Detection
python
Copy
Edit
height, width, _ = frame.shape
blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
yolo_net.setInput(blob)
detections = yolo_net.forward(output_layers)
Converts the frame into a YOLO-compatible format (blobFromImage).

Runs YOLO detection to find faces.

8. Process YOLO Face Detections
python
Copy
Edit
boxes = []
confidences = []

for detection in detections:
    for obj in detection:
        scores = obj[5:]
        confidence = max(scores)
        if confidence > 0.4:  # Confidence threshold for face detection
            center_x, center_y, w, h = (obj[:4] * [width, height, width, height]).astype(int)
            x, y = int(center_x - w / 2), int(center_y - h / 2)
            boxes.append([x, y, w, h])
            confidences.append(float(confidence))
Extracts face bounding boxes from the YOLO output.

Faces must have a confidence score above 0.4 (adjustable for better results).

python
Copy
Edit
indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
Non-Maximum Suppression (NMS) removes overlapping detections.

9. Preprocess the Face for CNN Models
python
Copy
Edit
if len(indices) > 0:
    for i in indices.flatten():
        x, y, w, h = boxes[i]
        face = frame[y:y + h, x:x + w]

        if face.shape[0] == 0 or face.shape[1] == 0:
            continue

        # ✅ Histogram Equalization for better detection
        face_gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        face_gray = cv2.equalizeHist(face_gray)  # Improves contrast
        face_gray = cv2.resize(face_gray, (224, 224))
        face_gray = np.stack([face_gray] * 3, axis=-1) / 255.0  # Normalize
        face_gray = np.expand_dims(face_gray, axis=0)
Converts the face to grayscale.

Applies histogram equalization (improves contrast, enhances details).

Resizes to 224x224 pixels.

Converts to RGB format and normalizes pixel values (/255.0).

10. Predict Emotion Using the CNN Model
python
Copy
Edit
emotion_preds = emotion_model.predict(face_gray)
emotion_label = np.argmax(emotion_preds)
emotion_history.append(emotion_label)

# ✅ Smooth Emotion Prediction
final_emotion_label = max(set(emotion_history), key=emotion_history.count)
Gets the emotion prediction from emotion_model.

Uses a rolling average (emotion_history) to smooth predictions and prevent flickering.

11. Predict Activity Every 10 Seconds
python
Copy
Edit
if time.time() - last_activity_time > 10:
    activity_preds = activity_model.predict(face_gray)
    activity_label = np.argmax(activity_preds)
    last_activity_time = time.time()
Prevents the activity prediction from changing every frame.

Runs activity detection every 10 seconds.

12. Draw Predictions on the Video Frame
python
Copy
Edit
cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
cv2.putText(frame, f"Emotion: {emotion_labels[final_emotion_label]}", (x, y - 10), 
            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
cv2.putText(frame, f"Activity: {activity_labels[activity_label]}", (x, y - 30), 
            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
Draws a bounding box around the detected face.

Displays the predicted emotion and activity on the screen.

13. Display the Final Output
python
Copy
Edit
cv2.imshow('Emotion & Activity Detection', frame)

if cv2.waitKey(1) & 0xFF == ord('q'):
    break
Shows the video output with real-time predictions.

Press "Q" to exit.

Final Thoughts
✅ Face Detection is Faster (uses YOLO with GPU).
✅ Emotion Prediction is More Stable (rolling average prevents flickering).
✅ Activity Prediction is Less Frequent (runs every 10 seconds to improve accuracy).
✅ Better Face Processing (histogram equalization enhances details).


this is explanation forr yolo model 
