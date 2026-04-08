import cv2
import numpy as np
import pickle
import time
import os
from threading import Lock
import mediapipe as mp

try:
    import tflite_runtime.interpreter as tflite

    TFLITE_AVAILABLE = True
    print("Using tflite_runtime")
except ImportError:
    try:
        import tensorflow as tf

        TFLITE_AVAILABLE = False
        print("Using full TensorFlow with Lite support")
    except ImportError:
        print("Neither TensorFlow nor tflite_runtime available")
        TFLITE_AVAILABLE = None


class FaceNetRecognition:
    def __init__(self):
        print("Initializing face recognition system...")

        self.model_path = "facenet.tflite"
        self.model_available = False
        self.TFLITE_AVAILABLE = TFLITE_AVAILABLE

        if os.path.exists(self.model_path):
            print(f"Found FaceNet TensorFlow Lite: {self.model_path}")
            self.facenet_model = self.load_facenet()
            self.model_available = self.facenet_model is not None
        else:
            print(f"FaceNet TensorFlow Lite not found at {self.model_path}. Using simplified recognition.")
            self.facenet_model = None
            self.model_available = False

        self.mp_face_detection = mp.solutions.face_detection
        self.face_detection = self.mp_face_detection.FaceDetection(
            model_selection=0,
            min_detection_confidence=0.5
        )

        self.operator_embeddings = []
        self.face_data_lock = Lock()
        self.learning_mode = False

        self.capture_interval = 0.15
        self.max_samples = 400
        self.last_capture_time = 0

    def load_facenet(self):
        try:
            print("Loading FaceNet TensorFlow Lite model...")

            if self.TFLITE_AVAILABLE:
                interpreter = tflite.Interpreter(model_path=self.model_path)
            else:
                interpreter = tf.lite.Interpreter(model_path=self.model_path)

            interpreter.allocate_tensors()
            print("FaceNet TensorFlow Lite loaded successfully!")

            return interpreter
        except Exception as e:
            print(f"Error loading FaceNet TensorFlow Lite: {e}")
            return None

    def get_face_embedding(self, face_image):
        if not self.model_available:
            print("Using simple embedding (TFLite model not available)")
            return self.get_simple_embedding(face_image)

        try:
            input_details = self.facenet_model.get_input_details()
            input_shape = input_details[0]['shape']

            target_height, target_width = input_shape[1], input_shape[2]

            face_resized = cv2.resize(face_image, (target_width, target_height))
            face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)

            if input_details[0]['dtype'] == np.float32:
                face_normalized = (face_rgb.astype('float32') - 127.5) / 128.0
            else:
                face_normalized = face_rgb.astype('uint8')

            face_batch = np.expand_dims(face_normalized, axis=0)

            self.facenet_model.set_tensor(input_details[0]['index'], face_batch)

            self.facenet_model.invoke()

            output_details = self.facenet_model.get_output_details()
            embedding = self.facenet_model.get_tensor(output_details[0]['index'])

            embedding = embedding.flatten()
            
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm

            return embedding

        except Exception as e:
            print(f"FaceNet TensorFlow Lite error, using fallback: {e}")
            return self.get_simple_embedding(face_image)

    def detect_faces(self, frame):
        try:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_detection.process(rgb_frame)

            faces = []
            if results.detections:
                for detection in results.detections:
                    bboxC = detection.location_data.relative_bounding_box
                    h, w, _ = frame.shape

                    x = int(bboxC.xmin * w)
                    y = int(bboxC.ymin * h)
                    width = int(bboxC.width * w)
                    height = int(bboxC.height * h)

                    margin_x = int(width * 0.1)
                    margin_y = int(height * 0.1)

                    x = max(0, x - margin_x)
                    y = max(0, y - margin_y)
                    width = min(w - x, width + 2 * margin_x)
                    height = min(h - y, height + 2 * margin_y)

                    faces.append((x, y, width, height))

            return faces
        except Exception as e:
            print(f"MediaPipe face detection error: {e}")
            return []

    def extract_face(self, frame, x, y, w, h):
        try:
            h_frame, w_frame = frame.shape[:2]
            x = max(0, x)
            y = max(0, y)
            w = min(w, w_frame - x)
            h = min(h, h_frame - y)

            if w > 0 and h > 0:
                return frame[y:y + h, x:x + w]
            else:
                return np.array([])
        except Exception as e:
            print(f"Face extraction error: {e}")
            return np.array([])

    def recognize_operator(self, face_embedding):
        if len(self.operator_embeddings) < 5:
            return False, 0.0

        centroid = np.mean(self.operator_embeddings, axis=0)
        centroid_similarity = self._cosine_similarity(face_embedding, centroid)

        if centroid_similarity <= 0.60:
            return False, centroid_similarity

        similarities = [self._cosine_similarity(face_embedding, emb)
                        for emb in self.operator_embeddings]

        confirming_matches = sum(1 for s in similarities if s > 0.60)
        match_ratio = confirming_matches / len(similarities)

        is_operator = centroid_similarity > 0.60 and match_ratio > 0.20

        return is_operator, centroid_similarity

    def get_augmented_embeddings(self, face_embedding, num_variants=3):
        augmented_embeddings = []

        augmented_embeddings.append(face_embedding)

        for i in range(num_variants):
            noise_level = 0.01 + (i * 0.005)
            noise = np.random.normal(0, noise_level, face_embedding.shape)
            augmented_embedding = face_embedding + noise
            augmented_embeddings.append(augmented_embedding)

        return augmented_embeddings

    def capture_operator_face(self, face_embedding):
        current_time = time.time()
        if current_time - self.last_capture_time >= self.capture_interval:
            with self.face_data_lock:
                augmented_embeddings = self.get_augmented_embeddings(face_embedding, 3)

                for aug_emb in augmented_embeddings:
                    if len(self.operator_embeddings) < self.max_samples:
                        self.operator_embeddings.append(aug_emb)

                self.last_capture_time = current_time
                progress = len(self.operator_embeddings) / self.max_samples * 100
                print(f"Progress: {len(self.operator_embeddings)}/{self.max_samples} ({progress:.0f}%)")

                if len(self.operator_embeddings) >= self.max_samples:
                    self.learning_mode = False
                    print("AUTOMATIC LEARNING COMPLETION!")
                    self.check_embedding_quality()
                    return True
            return False

    def check_embedding_quality(self):
        if len(self.operator_embeddings) < 5:
            return

        centroid = np.mean(self.operator_embeddings, axis=0)
        similarities_to_centroid = []

        for emb in self.operator_embeddings:
            similarity = np.dot(emb, centroid) / (np.linalg.norm(emb) * np.linalg.norm(centroid))
            similarities_to_centroid.append(similarity)

        avg_similarity = np.mean(similarities_to_centroid)
        min_similarity = np.min(similarities_to_centroid)

        print(f"Embedding quality: {avg_similarity:.3f} (min: {min_similarity:.3f})")

        if avg_similarity < 0.7:
            print("Warning: Low embedding consistency!")

    def start_learning_operator(self):
        with self.face_data_lock:
            self.learning_mode = True
            self.operator_embeddings = []
            self.last_capture_time = 0
            
    def _cosine_similarity(self, embedding1, embedding2):
        return np.dot(embedding1, embedding2)

    def get_simple_embedding(self, face_image):
        try:
            face_resized = cv2.resize(face_image, (128, 128))
            face_gray = cv2.cvtColor(face_resized, cv2.COLOR_BGR2GRAY)
            hist = cv2.calcHist([face_gray], [0], None, [32], [0, 256]).flatten()
            features = face_gray.flatten().astype('float32') / 255.0
            embedding = np.concatenate([hist, features[:224]])
            
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm
            
            return embedding
        except Exception as e:
            print(f"Simple embedding error: {e}")
            return np.random.randn(256)

    def forget_operator(self):
        with self.face_data_lock:
            self.operator_embeddings = []

    def __del__(self):
        if hasattr(self, 'face_detection'):
            self.face_detection.close()

"""
    def save_operator(self, filename='operator.pkl'):
        try:
            with self.face_data_lock:
                if self.operator_embeddings:
                    model_data = {'operator_embeddings': self.operator_embeddings}
                    with open(filename, 'wb') as f:
                        pickle.dump(model_data, f)
                    print(f"Operator data saved to {filename}")
        except Exception as e:
            print(f"Save error: {e}")
"""
