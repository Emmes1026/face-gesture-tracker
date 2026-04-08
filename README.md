# face-gesture-tracker
Python-based software for an automatic face tracking camera with hand gesture recognition.

# Description
This project was developed as part of a bachelor's degree thesis. The main premise was to utilize available tools to integrate advanced AI face and hand gesture recognition in tandem with edge computing devices.

The system relies on MediaPipe for detecting human faces and hands, FaceNet for recognizing specific individuals, and Adafruit libraries for precise servo motor control.

# Key Features
* Real-time face and hands detection and tracking pre-trained models.
* Hand gesture recognition.
* Hardware-level Pan/Tilt camera control.

[![Watch the video](https://img.youtube.com/vi/9OMbF3YzcZI/hqdefault.jpg)](https://youtu.be/9OMbF3YzcZI)

# Technicalities
This project is designed and optimized specifically for the Raspberry Pi 5. On startup, the system automatically verifies the availability of the camera and the appropriate machine learning models. Because the system is built for edge computing, it exclusively utilizes TensorFlow Lite (.tflite) models.

For development and debugging purposes, the program can also be executed on Windows. In this environment, actual TFLite inference is bypassed. Instead, the system generates random mock embeddings to simulate AI output and prevent crashes, allowing developers to test the core logic without needing the physical edge hardware.

# 3D enclosure
**[Click here to view and download the 3D Enclosure files](https://github.com/Emmes1026/auto-face-tracker-enclosure/tree/main)**


