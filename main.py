import cv2
import time
import os
from datetime import datetime
from FaceRecognition import FaceNetRecognition
from GestRecognition import HandGestureRecognizer, Gesture

from ServoTracker import ServoTracker, GPIOZERO_AVAILABLE


class IntegratedRecognitionApp:
    def __init__(self):
        self.face_recognizer = FaceNetRecognition()
        self.gesture_recognizer = HandGestureRecognizer()

        self.action_triggered = False
        self.snapshot_feedback_timer = 0
        self.learning_target_last_pos = None

        self.recognition_interval = 5
        self.frame_count = 0
        self.last_identities = {}

        self.current_mode = "RECOGNITION"
        self.tracking_target = "FACE"  # FACE / FINGER / STOP
        self.tracked_hand_side = None

        self.active_hold_gesture = None
        self.active_hold_hand = None
        self.hold_start_time = 0
        self.HOLD_THRESHOLD = 1.0
        self.action_triggered = False

        self.snapshot_feedback_timer = 0

        self.pan_channel = 0
        self.tilt_channel = 4
        self.tracker = None

        self.picam2 = None
        self.cap = None
        self.is_pi_camera = False


        if not os.path.exists('snapshots'):
            os.makedirs('snapshots')

        try:
            from picamera2 import Picamera2 # type: ignore
            self.picam2 = Picamera2()

            config = self.picam2.create_preview_configuration(
                main={"size": (820, 616), "format": "BGR888"},
                sensor={"output_size": (1640, 1232)}
            )
            self.picam2.configure(config)
            self.picam2.start()

            self.is_pi_camera = True
            self.frame_width = 820
            self.frame_height = 616
        except (ImportError, ModuleNotFoundError, RuntimeError):
            self.frame_width = 1280
            self.frame_height = 720
            print("picamera2 not found, using USB camera")
            self.is_pi_camera = False
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                raise IOError("Error while using USB camera")
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)


        if GPIOZERO_AVAILABLE:
            self.tracker = ServoTracker(self.pan_channel, self.tilt_channel)

    def run(self):
        print("Integrated Face and Gesture Recognition")

        gesture_recognition_enabled = True

        if self.is_pi_camera: self.picam2.start()
        try:
            while True:
                frame = None
                if self.is_pi_camera:
                    frame = self.picam2.capture_array()
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                else:
                    ret, frame = self.cap.read()
                    if not ret: break

                if frame is None: break
                frame = cv2.flip(frame, 1)

                clean_frame = frame.copy()

                operator_face_center = None
                center_x = self.frame_width // 2
                center_y = self.frame_height // 2

                if self.tracker is not None:
                    dz = self.tracker.deadcenter
                    cv2.rectangle(frame, (center_x - dz, center_y - dz), (center_x + dz, center_y + dz), (128, 128, 128), 2)

                faces = self.face_recognizer.detect_faces(frame)

                if not faces:
                    self.learning_target_last_pos = None

                else:
                    faces_with_props = []
                    for rect in faces:
                        (x, y, w, h) = rect
                        cx, cy = x + w / 2.0, y + h / 2.0
                        faces_with_props.append({'rect': rect, 'center': (cx, cy)})

                    target_face_index = -1

                    if self.current_mode == "LEARNING":
                        if self.learning_target_last_pos is None:
                            min_dist = float('inf')
                            for i, face in enumerate(faces_with_props):
                                d = ((face['center'][0] - center_x) ** 2 + (face['center'][1] - center_y) ** 2) ** 0.5
                                if d < min_dist:
                                    min_dist = d
                                    target_face_index = i
                        else:
                            last_x, last_y = self.learning_target_last_pos
                            min_dist = float('inf')

                            for i, face in enumerate(faces_with_props):
                                d = ((face['center'][0] - last_x) ** 2 + (face['center'][1] - last_y) ** 2) ** 0.5
                                if d < min_dist:
                                    min_dist = d
                                    target_face_index = i

                            if min_dist > 200:
                                target_face_index = -1
                                self.learning_target_last_pos = None

                                min_dist_center = float('inf')
                                for i, face in enumerate(faces_with_props):
                                    d = ((face['center'][0] - center_x) ** 2 + (face['center'][1] - center_y) ** 2) ** 0.5
                                    if d < min_dist_center:
                                        min_dist_center = d
                                        target_face_index = i

                        if target_face_index != -1:
                            self.learning_target_last_pos = faces_with_props[target_face_index]['center']


                    if self.frame_count % 100 == 0: self.last_identities = {}

                    should_recognize = (self.frame_count % self.recognition_interval == 0)

                    for i, face_data in enumerate(faces_with_props):
                        (x, y, w, h) = face_data['rect']
                        cx, cy = face_data['center']

                        face_color = (255, 255, 255)
                        face_status = ""
                        is_current_operator = False

                        if self.current_mode == "LEARNING":
                            face_img = self.face_recognizer.extract_face(frame, x, y, w, h)
                            is_target = (i == target_face_index)

                            if is_target:
                                face_color = (0, 255, 255)
                                face_status = "LOCKED TARGET"

                                if face_img.size > 0:
                                    embedding = self.face_recognizer.get_face_embedding(face_img)
                                    if self.face_recognizer.capture_operator_face(embedding):
                                        self.current_mode = "RECOGNITION"
                                        self.learning_target_last_pos = None
                            else:
                                face_color = (80, 80, 80)
                                face_status = "IGNORED"

                        else:

                            if not should_recognize and i in self.last_identities:
                                face_status, face_color, is_current_operator = self.last_identities[i]
                            else:
                                face_img = self.face_recognizer.extract_face(frame, x, y, w, h)

                                if face_img.size > 0:
                                    embedding = self.face_recognizer.get_face_embedding(face_img)
                                    is_operator, confidence = self.face_recognizer.recognize_operator(embedding)

                                    if is_operator:
                                        face_color = (0, 255, 0)
                                        face_status = f"OPERATOR ({confidence:.2f})"
                                        is_current_operator = True
                                    else:
                                        face_color = (0, 0, 255)
                                        face_status = f"UNKNOWN ({confidence:.2f})"
                                        is_current_operator = False

                                    self.last_identities[i] = (face_status, face_color, is_current_operator)
                                else:
                                    face_color = (128, 128, 128)
                                    is_current_operator = False

                            if is_current_operator:
                                operator_face_center = (cx, cy)

                        cv2.rectangle(frame, (x, y), (x + w, y + h), face_color, 2)
                        cv2.putText(frame, face_status, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, face_color, 2)

                servo_target_x, servo_target_y = None, None
                dominant_gesture = None
                dominant_hand_side = None

                if gesture_recognition_enabled:
                    gestures, hand_landmarks_list, handedness_list = self.gesture_recognizer.detect_gestures(frame)

                    if gestures:
                        self.gesture_recognizer.draw_hand_landmarks(frame, hand_landmarks_list)

                        hand_candidates = []

                        for i, (gesture, landmarks, hand_side) in enumerate(
                                zip(gestures, hand_landmarks_list, handedness_list)):
                            self.gesture_recognizer.draw_hand_info(frame, landmarks, gesture, hand_side)

                            wrist = landmarks.landmark[0]
                            hand_x = wrist.x * self.frame_width
                            hand_y = wrist.y * self.frame_height
                            dist_score = 999999

                            if (
                                    self.tracking_target == "FACE" or self.tracking_target == "STOP") and operator_face_center is not None:
                                face_x, face_y = operator_face_center
                                dist_score = ((hand_x - face_x) ** 2 + (hand_y - face_y) ** 2) ** 0.5
                            elif self.tracking_target == "FINGER":
                                dist_score = ((hand_x - center_x) ** 2 + (hand_y - center_y) ** 2) ** 0.5
                            else:
                                dist_score = ((hand_x - center_x) ** 2 + (hand_y - center_y) ** 2) ** 0.5

                            hand_candidates.append({
                                'gesture': gesture,
                                'landmarks': landmarks,
                                'hand_side': hand_side,
                                'score': dist_score
                            })

                        if hand_candidates:
                            hand_candidates.sort(key=lambda x: x['score'])
                            best_hand = hand_candidates[0]

                            gesture = best_hand['gesture']
                            landmarks = best_hand['landmarks']
                            hand_side = best_hand['hand_side']

                            is_command_gesture = False
                            if gesture in [Gesture.THUMBS_UP, Gesture.THUMBS_DOWN, Gesture.POINTING,
                                           Gesture.CLOSED_FIST, Gesture.OPEN_HAND, Gesture.VICTORY]:
                                if self.tracking_target == "FINGER":
                                    if gesture == Gesture.CLOSED_FIST:
                                        is_command_gesture = True
                                    else:
                                        is_command_gesture = False
                                else:
                                    is_command_gesture = True

                            if is_command_gesture:
                                dominant_gesture = gesture
                                dominant_hand_side = hand_side

                            if self.tracking_target == "FINGER" and self.tracked_hand_side == hand_side:
                                tip_x = int(landmarks.landmark[8].x * self.frame_width)
                                tip_y = int(landmarks.landmark[8].y * self.frame_height)
                                servo_target_x, servo_target_y = tip_x, tip_y
                                cv2.circle(frame, (tip_x, tip_y), 10, (255, 0, 255), -1)

                self._process_gesture_hold(frame, clean_frame, dominant_gesture, dominant_hand_side)

                if self.tracker is not None:
                    final_target_x = None
                    final_target_y = None
                    cv2.rectangle(frame, (center_x - dz, center_y - dz), (center_x + dz, center_y + dz), (128, 128, 128), 2)

                    if self.tracking_target == "STOP":
                        pass
                    elif self.tracking_target == "FINGER":
                        if servo_target_x is not None:
                            final_target_x = servo_target_x
                            final_target_y = servo_target_y
                    elif self.tracking_target == "FACE":
                        if operator_face_center is not None:
                            final_target_x = operator_face_center[0]
                            final_target_y = operator_face_center[1]

                    if final_target_x is not None:
                        self.tracker.update(final_target_x, final_target_y, self.frame_width, self.frame_height)

                mode_text = f"Sys: {self.current_mode}"
                track_status = self.tracking_target
                track_color = (255, 255, 255)
                if self.tracking_target == "STOP":
                    track_status = "!!! STOPPED !!!"
                    track_color = (0, 0, 255)
                elif self.tracking_target == "FINGER":
                    track_status += f" ({self.tracked_hand_side})"

                cv2.putText(frame, mode_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(frame, f"Track: {track_status}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, track_color, 2)

                if time.time() - self.snapshot_feedback_timer < 2.0:
                    cv2.putText(frame, "SNAPSHOT SAVED!", (self.frame_width // 2 - 100, self.frame_height // 2),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3)

                help_y = frame.shape[0] - 20
                cv2.putText(frame, "Fist=Track face | Point=Track finger |  Open=Stop | Thumb up=Learn | Thumb down=Forget  | V=Photo", (10, help_y), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (200, 200, 200), 1)

                cv2.imshow('Integrated Recognition', frame)

                self.frame_count += 1

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('g'):
                    gesture_recognition_enabled = not gesture_recognition_enabled
        finally:
            if self.is_pi_camera:
                self.picam2.stop()
            else:
                self.cap.release()
            if self.tracker is not None: self.tracker.cleanup()
            cv2.destroyAllWindows()

    def _process_gesture_hold(self, frame, clean_frame, detected_gesture, hand_side):
        current_time = time.time()

        if detected_gesture is not None:
            if (detected_gesture == self.active_hold_gesture and
                    hand_side == self.active_hold_hand):

                if not self.action_triggered:
                    elapsed_time = current_time - self.hold_start_time
                    remaining_time = max(0.0, self.HOLD_THRESHOLD - elapsed_time)

                    progress_text = f"HOLD: {detected_gesture.name} {remaining_time:.1f}s"
                    cv2.putText(frame, progress_text, (10, 450),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)

                    if elapsed_time >= self.HOLD_THRESHOLD:
                        self._execute_hold_action(clean_frame, detected_gesture, hand_side)
                        self.action_triggered = True
            else:
                self.active_hold_gesture = detected_gesture
                self.active_hold_hand = hand_side
                self.hold_start_time = current_time
                self.action_triggered = False
        else:
            self.active_hold_gesture = None
            self.active_hold_hand = None
            self.action_triggered = False

    def _execute_hold_action(self, clean_frame, gesture, hand_side):
        print(f"ACTION TRIGGERED: {gesture.name}")

        if gesture == Gesture.THUMBS_UP:
            print(">>> STARTING LEARNING MODE <<<")
            self.start_learning_operator()

        elif gesture == Gesture.THUMBS_DOWN:
            print(">>> FORGETTING OPERATOR <<<")
            self.forget_operator()

        elif gesture == Gesture.POINTING:
            print(f">>> TRACKING FINGER ({hand_side}) <<<")
            self.tracking_target = "FINGER"
            self.tracked_hand_side = hand_side

        elif gesture == Gesture.OPEN_HAND:
            print(">>> CAMERA STOPPED/FROZEN <<<")
            self.tracking_target = "STOP"
            self.tracked_hand_side = None

        elif gesture == Gesture.CLOSED_FIST:
            print(">>> REVERT TO FACE TRACKING <<<")
            self.tracking_target = "FACE"
            self.tracked_hand_side = None

        elif gesture == Gesture.VICTORY:
            print(">>> TAKING SNAPSHOT <<<")
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"snapshots/img_{timestamp}.jpg"

            cv2.imwrite(filename, clean_frame)
            print(f"Saved CLEAN image: {filename}")

            self.snapshot_feedback_timer = time.time()

    def start_learning_operator(self):
        self.current_mode = "LEARNING"
        self.face_recognizer.start_learning_operator()

    def forget_operator(self):
        self.face_recognizer.forget_operator()


if __name__ == "__main__":
    try:
        app = IntegratedRecognitionApp()
        app.run()
    except Exception as e:
        print(f"Błąd: {e}")
