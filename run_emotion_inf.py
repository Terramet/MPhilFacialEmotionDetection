import cv2
import numpy as np
from tensorflow.lite.python.interpreter import Interpreter

# Load YOLO model
net = cv2.dnn.readNet("/models/yolov4_tiny_face_final.weights", "/models/yolov4-tiny_face.cfg")
layer_names = net.getUnconnectedOutLayersNames()

# Load TFLite model for emotion prediction
emotion_model_path = "models/emotion_model_MobileNetV2_FERPlus_8_Emotions_ofp.tflite"  # Path to your emotion model
interpreter = Interpreter(model_path=emotion_model_path)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Define label mapping for emotion predictions
label_mapping = {
    0: 'happpy',
    1: 'neutral',
    2: 'fear',
    3: 'surprise',
    4: 'sad',
    5: 'anger',
    6: 'disgust',
    7: 'contempt'
}

# Define the video capture object
cap = cv2.VideoCapture(0)

# Function to preprocess face for emotion prediction
def preprocess_face(face):
    face_resized = cv2.resize(face, (48, 48))  # Resize face to 48x48, depending on the model input size
    face_resized = cv2.resize(face, (224, 224))  # Resize face to 48x48, depending on the model input size
    #face_resized = face_resized.astype('float32') / 255.0  # Normalize to [0, 1] if using VGG16
    pre_img = cv2.cvtColor(face_resized, cv2.COLOR_BGR2GRAY)
    input_data = preprocess_frame(pre_img, [1,224,224])
    input_data = np.stack((input_data,)*3, axis=-1)  # Shape: (1, 224, 224, 3)
    return input_data

def preprocess_frame(frame, input_shape):
    frame_resized = cv2.resize(frame, (input_shape[2], input_shape[1]))
    input_data = np.expand_dims(frame_resized, axis=0).astype(np.float32)
    return input_data

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    height, width, channels = frame.shape

    # Prepare the image for YOLO
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(layer_names)

    # Initialize lists to hold detection data
    class_ids = []
    confidences = []
    boxes = []

    # Process each detection
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:  # Filter weak detections
                # Object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Apply Non-Maximum Suppression to suppress overlapping boxes
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    # Draw bounding boxes on the frame and predict emotion
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(class_ids[i])
            confidence = confidences[i]
            color = (0, 255, 0)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, f"Face: {confidence:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # Extract face for emotion recognition
            face = frame[y:y + h, x:x + w]
            if face.size > 0:
                # Preprocess the face and make emotion prediction
                face_input = preprocess_face(face)
                interpreter.set_tensor(input_details[0]['index'], face_input)
                interpreter.invoke()
                emotion_prediction = interpreter.get_tensor(output_details[0]['index'])[0]
                predicted_emotion = np.argmax(emotion_prediction)

                # Get emotion label
                emotion_label = label_mapping[predicted_emotion]
                cv2.putText(frame, f"Emotion: {emotion_label}", (x, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # Display the resulting frame
    cv2.imshow('Frame', frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close windows
cap.release()
cv2.destroyAllWindows()
