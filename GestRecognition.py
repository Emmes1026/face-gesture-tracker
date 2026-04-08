import cv2
import numpy as np
import mediapipe as mp
from enum import Enum


class Gesture(Enum):
    UNKNOWN = 0
    OPEN_HAND = 1
    CLOSED_FIST = 2
    THUMBS_UP = 3
    THUMBS_DOWN = 4
    VICTORY = 5
    POINTING = 6


class HandGestureRecognizer:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )

    def detect_gestures(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)

        gestures = []
        hand_landmarks_list = []
        handedness_list = []

        if results.multi_hand_landmarks:
            for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
                landmarks = []
                for landmark in hand_landmarks.landmark:
                    landmarks.append([landmark.x, landmark.y, landmark.z])
                landmarks = np.array(landmarks)

                handedness = "right"  # domyślnie
                if results.multi_handedness:
                    handedness = results.multi_handedness[i].classification[0].label.lower()

                gesture = self._classify_gesture(landmarks, handedness)
                gestures.append(gesture)
                hand_landmarks_list.append(hand_landmarks)
                handedness_list.append(handedness)

        return gestures, hand_landmarks_list, handedness_list

    def _classify_gesture(self, landmarks, handedness):
        if self._is_thumbs_up(landmarks, handedness):
            return Gesture.THUMBS_UP
        elif self._is_thumbs_down(landmarks, handedness):
            return Gesture.THUMBS_DOWN
        elif self._is_victory(landmarks):
            return Gesture.VICTORY
        elif self._is_pointing(landmarks):
            return Gesture.POINTING
        elif self._is_open_hand(landmarks):
            return Gesture.OPEN_HAND
        elif self._is_closed_fist(landmarks):
            return Gesture.CLOSED_FIST
        else:
            return Gesture.UNKNOWN

    def _is_open_hand(self, landmarks):
        finger_tips = [8, 12, 16, 20]
        finger_mcp = [5, 9, 13, 17]

        for tip, mcp in zip(finger_tips, finger_mcp):
            if landmarks[tip][1] > landmarks[mcp][1]:
                return False
        return True

    def _is_closed_fist(self, landmarks):
        finger_tips = [8, 12, 16, 20]
        finger_pip = [6, 10, 14, 18]

        for tip, pip in zip(finger_tips, finger_pip):
            if landmarks[tip][1] < landmarks[pip][1]:
                return False

        return True

    def _is_thumbs_up(self, landmarks, handedness):
        thumb_tip = landmarks[4]
        thumb_ip = landmarks[3]
        thumb_mcp = landmarks[2]
        wrist = landmarks[0]
        index_mcp = landmarks[5]

        if thumb_tip[1] >= index_mcp[1]:
            return False

        if not self._are_other_fingers_mostly_closed(landmarks):
            return False

        if handedness == "right":
            thumb_is_outward = thumb_tip[0] < thumb_mcp[0]
        else:
            thumb_is_outward = thumb_tip[0] > thumb_mcp[0]

        if not thumb_is_outward:
            return False

        thumb_is_straight = thumb_tip[1] < thumb_ip[1] - 0.02

        return thumb_is_straight

    def _is_thumbs_down(self, landmarks, handedness):
        thumb_tip = landmarks[4]
        thumb_ip = landmarks[3]
        thumb_mcp = landmarks[2]
        wrist = landmarks[0]

        if not self._are_other_fingers_mostly_closed(landmarks):
            return False

        if handedness == "right":
            thumb_on_correct_side = thumb_tip[0] < thumb_mcp[0]
        else:
            thumb_on_correct_side = thumb_tip[0] > thumb_mcp[0]

        thumb_below_wrist = thumb_tip[1] > wrist[1]

        thumb_extended_down = thumb_tip[1] > thumb_ip[1] + 0.02  # margines

        return thumb_on_correct_side and thumb_below_wrist and thumb_extended_down

    def _are_other_fingers_mostly_closed(self, landmarks):
        finger_tips = [8, 12, 16, 20]
        finger_pips = [6, 10, 14, 18]

        extended_count = 0
        for tip, pip in zip(finger_tips, finger_pips):
            if landmarks[tip][1] < landmarks[pip][1]:
                extended_count += 1

        return extended_count <= 1

    def _is_victory(self, landmarks):
        index_tip = landmarks[8]
        middle_tip = landmarks[12]
        ring_tip = landmarks[16]
        pinky_tip = landmarks[20]

        index_extended = index_tip[1] < landmarks[6][1]
        middle_extended = middle_tip[1] < landmarks[10][1]
        ring_closed = ring_tip[1] > landmarks[14][1]
        pinky_closed = pinky_tip[1] > landmarks[18][1]

        return index_extended and middle_extended and ring_closed and pinky_closed

    def _is_pointing(self, landmarks):
        index_tip = landmarks[8]
        middle_tip = landmarks[12]
        ring_tip = landmarks[16]
        pinky_tip = landmarks[20]

        index_extended = index_tip[1] < landmarks[6][1]
        other_fingers_closed = (middle_tip[1] > landmarks[10][1] and
                                ring_tip[1] > landmarks[14][1] and
                                pinky_tip[1] > landmarks[18][1])

        return index_extended and other_fingers_closed

    def draw_hand_landmarks(self, frame, hand_landmarks_list):
        for hand_landmarks in hand_landmarks_list:
            self.mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                self.mp_hands.HAND_CONNECTIONS,
                self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                self.mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2)
            )

    def draw_hand_info(self, frame, hand_landmarks, gesture, handedness):
        h, w = frame.shape[:2]

        thumb_tip = hand_landmarks.landmark[4]
        thumb_tip_px = (int(thumb_tip.x * w), int(thumb_tip.y * h))

        info_text = f"{gesture.name} ({handedness})"
        text_color = (0, 255, 0) if gesture in [Gesture.THUMBS_UP, Gesture.THUMBS_DOWN] else (0, 255, 255)
        cv2.putText(frame, info_text, (thumb_tip_px[0] + 15, thumb_tip_px[1]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2)