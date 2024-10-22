# Real-Time Face Detection and Emotion Recognition using YOLO and TFLite

## Overview

This project uses a pre-trained YOLO model for real-time face detection and a TensorFlow Lite (TFLite) model to perform emotion recognition from the detected faces. It captures video from the camera, detects faces using YOLO, and then predicts the emotion of the detected face using the MobileNetV2-based emotion recognition model trained on FERPlus.

## Prerequisites

1. **Python 3.x**: Ensure that Python 3.x is installed.
2. **OpenCV**: Used for real-time computer vision (face detection and drawing bounding boxes).
3. **NumPy**: Used for numerical operations and data manipulation.
4. **TensorFlow Lite**: Required to load and run the emotion recognition model.
5. **YOLO Weights and Config Files**: Pre-trained YOLO weights and configuration for face detection.

## Installation

Install the required Python packages:

```bash
pip install opencv-python numpy tensorflow
```

You also need:
- YOLO weights (eg `yolov4_tiny_face_final.weights`) and configuration file (eg `yolov4-tiny_face.cfg`).
- TensorFlow Lite model for emotion recognition (eg `emotion_model_MobileNetV2_FERPlus_8_Emotions_ofp.tflite`).

## How to Run

**If running on a robot install tflite-runtime instead and change the** from tensorflow.lite.python.interpreter import Interpreter **import to:** from tflite_runtime.interpreter import Interpreter

1. **Download YOLO Model**:
   - Download the YOLO face detection model weights (`yolov4_tiny_face_final.weights`) and configuration file (`yolov4-tiny_face.cfg`).

2. **Prepare Emotion Detection Model**:
   - Download or use a pre-trained TFLite model for emotion recognition. The code assumes the model path is:
     
     ```python
     "D:\\models\\emotion_model_MobileNetV2_FERPlus_8_Emotions_ofp.tflite"
     ```
   
   - You can modify this path as needed.

3. **Run the Code**:
   To run the code, execute the following command in your terminal:
   
   ```bash
   python emotion_detection.py
   ```

## Emotion Mapping

The TFLite model predicts one of the following emotions:
- 0: happy
- 1: neutral
- 2: fear
- 3: surprise
- 4: sad
- 5: anger
- 6: disgust
- 7: contempt

## Code Explanation

1. **YOLO for Face Detection**:
   - The code uses the pre-trained YOLOv4-tiny model to detect faces in real time.
   - Bounding boxes are drawn around detected faces, and the confidence level is displayed.

2. **Emotion Recognition**:
   - Once a face is detected, it is preprocessed and passed to the MobileNetV2-based TFLite model to predict the emotion.
   - The predicted emotion label is displayed below the face in the video frame.

3. **Preprocessing**:
   - The detected face is resized to the input shape required by the emotion model (`224x224`).
   - The face is normalized and converted to grayscale before passing to the model.

## Dependencies

- **OpenCV**: Captures real-time video, processes frames, and displays the results.
- **NumPy**: Handles array manipulation and numerical operations.
- **TensorFlow Lite Interpreter**: Runs the emotion detection model in real time.

## Key Functions

- `preprocess_face(face)`: Prepares the detected face for emotion prediction by resizing, normalizing, and converting it to the appropriate format.
- `preprocess_frame(frame, input_shape)`: Resizes and preprocesses frames for model input.
  
## Output

- **Real-time face detection**: Bounding boxes are drawn around detected faces.
- **Emotion prediction**: The detected emotion is displayed alongside the face.

## Troubleshooting

- **No frame captured error**: Ensure your camera is correctly connected and accessible.
- **Model file errors**: Ensure that the correct path to the YOLO and emotion recognition models is provided in the code.

## Future Improvements

- Add support for multiple faces in a single frame.
- Implement more robust face preprocessing techniques.
- Enhance emotion prediction accuracy by using an ensemble of models.

## License

This project is open-source and free to use under the MIT License.

---

This project demonstrates how to use deep learning models (YOLO and MobileNetV2) for real-time face detection and emotion recognition. Feel free to experiment with different models and settings to improve performance!