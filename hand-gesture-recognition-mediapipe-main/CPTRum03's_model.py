#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hand Gesture Recognition for Drone Control
Updated version with modern libraries and improved code structure
"""

import csv
import copy
import argparse
import itertools
import time
import logging
from collections import Counter, deque
from pathlib import Path
from typing import Optional, List, Tuple, Dict, Any

import cv2 as cv
import numpy as np
import mediapipe as mp
from pymavlink import mavutil

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DroneController:
    """Handles MAVLink communication with drone/SITL"""
    
    def __init__(self, connection_string: str = 'udp:127.0.0.1:14550'):
        self.connection_string = connection_string
        self.master: Optional[mavutil.mavlink_connection] = None
        self.connect()
    
    def connect(self) -> bool:
        """Establish connection to SITL/drone"""
        try:
            self.master = mavutil.mavlink_connection(self.connection_string)
            logger.info(f"Connected to drone at {self.connection_string}")
            return True
        except Exception as e:
            logger.error(f"Connection failed: {e}")
            return False
    
    def set_rc_override(self, roll: Optional[int] = None, pitch: Optional[int] = None, 
                       throttle: Optional[int] = None, yaw: Optional[int] = None):
        """Override RC channels for manual control (1000-2000 PWM)"""
        if not self.master:
            logger.warning("No MAVLink connection available")
            return
            
        rc_channels = [65535] * 8  # Initialize all channels to "ignore"
        
        if roll is not None:
            rc_channels[0] = max(1000, min(2000, roll))
        if pitch is not None:
            rc_channels[1] = max(1000, min(2000, pitch))
        if throttle is not None:
            rc_channels[2] = max(1000, min(2000, throttle))
        if yaw is not None:
            rc_channels[3] = max(1000, min(2000, yaw))
        
        self.master.mav.rc_channels_override_send(
            self.master.target_system,
            self.master.target_component,
            *rc_channels[:8]
        )
    
    def execute_command(self, command: str, landmark_list: Optional[List] = None):
        """Execute drone command based on gesture recognition"""
        if not self.master:
            logger.error("No MAVLink connection")
            return False
        
        try:
            # Ensure heartbeat
            self.master.wait_heartbeat()
            
            command_map = {
                "Circle Clockwise": self._circle_command,
                "Circle Counter Clockwise": self._circle_command,
                "Arm": self._arm_command,
                "Disarm": self._disarm_command,
                "Takeoff": self._takeoff_command,
                "Return to Land": self._rtl_command,
                "Yaw Clockwise": self._yaw_command,
                "Yaw Counter Clockwise": self._yaw_command,
                "Follow": self._follow_command
            }
            
            if command in command_map:
                return command_map[command](command, landmark_list)
            else:
                logger.warning(f"Unknown command: {command}")
                return False
                
        except Exception as e:
            logger.error(f"Command execution failed: {e}")
            return False
    
    def _circle_command(self, command: str, landmark_list: Optional[List] = None):
        """Execute circle command"""
        if 'CIRCLE' not in self.master.mode_mapping():
            logger.error("CIRCLE mode not available")
            return False
        
        self.master.set_mode('CIRCLE')
        direction = 1 if "Clockwise" in command else -1
        
        self.master.mav.command_long_send(
            self.master.target_system,
            self.master.target_component,
            mavutil.mavlink.MAV_CMD_DO_GO_AROUND,
            0, 5, 2, direction, 0, 0, 0, 0
        )
        logger.info(f"Executed {command} (Radius:5m, Speed:2m/s)")
        return True
    
    def _arm_command(self, command: str, landmark_list: Optional[List] = None):
        """Arm the drone"""
        self.master.set_mode('GUIDED')
        self.master.mav.command_long_send(
            self.master.target_system,
            self.master.target_component,
            mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
            0, 1, 0, 0, 0, 0, 0, 0
        )
        logger.info("Arm command sent")
        return True
    
    def _disarm_command(self, command: str, landmark_list: Optional[List] = None):
        """Disarm the drone"""
        self.master.mav.command_long_send(
            self.master.target_system,
            self.master.target_component,
            mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
            0, 0, 0, 0, 0, 0, 0, 0
        )
        logger.info("Disarm command sent")
        return True
    
    def _takeoff_command(self, command: str, landmark_list: Optional[List] = None):
        """Execute takeoff command"""
        self.master.set_mode('GUIDED')
        self.master.mav.command_long_send(
            self.master.target_system,
            self.master.target_component,
            mavutil.mavlink.MAV_CMD_NAV_TAKEOFF,
            0, 0, 0, 0, 0, 0, 0, 15
        )
        logger.info("Takeoff command sent (15m)")
        return True
    
    def _rtl_command(self, command: str, landmark_list: Optional[List] = None):
        """Return to launch"""
        self.master.mav.command_long_send(
            self.master.target_system,
            self.master.target_component,
            mavutil.mavlink.MAV_CMD_NAV_RETURN_TO_LAUNCH,
            0, 0, 0, 0, 0, 0, 0, 0
        )
        logger.info("RTL command sent")
        return True
    
    def _yaw_command(self, command: str, landmark_list: Optional[List] = None):
        """Execute yaw command"""
        self.master.set_mode('GUIDED')
        direction = 1 if "Clockwise" in command else -1
        
        self.master.mav.command_long_send(
            self.master.target_system,
            self.master.target_component,
            mavutil.mavlink.MAV_CMD_CONDITION_YAW,
            0, 360, 30, direction, 1, 0, 0, 0
        )
        logger.info(f"Yaw command sent ({'CW' if direction == 1 else 'CCW'})")
        return True
    
    def _follow_command(self, command: str, landmark_list: Optional[List] = None):
        """Execute follow command using hand position"""
        if not landmark_list or len(landmark_list) < 9:
            logger.warning("No landmark data for follow command")
            return False
        
        # Use index fingertip (landmark 8) for control
        hand_x, hand_y = landmark_list[8][0], landmark_list[8][1]
        
        # Convert hand position to drone movement (adjust thresholds as needed)
        roll = pitch = throttle = yaw = None
        
        if hand_x < 300:  # Left
            roll = 1400
        elif hand_x > 600:  # Right
            roll = 1600
            
        if hand_y < 200:  # Up
            throttle = 1600
        elif hand_y > 400:  # Down
            throttle = 1400
        
        self.set_rc_override(roll=roll, pitch=pitch, throttle=throttle, yaw=yaw)
        logger.info("Follow mode active")
        return True


class HandGestureRecognizer:
    """Handles hand gesture recognition using MediaPipe"""
    
    def __init__(self, min_detection_confidence: float = 0.7, 
                 min_tracking_confidence: float = 0.5):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        
        # Load classifiers (assuming these classes exist)
        try:
            from model import KeyPointClassifier, PointHistoryClassifier
            self.keypoint_classifier = KeyPointClassifier()
            self.point_history_classifier = PointHistoryClassifier()
        except ImportError:
            logger.warning("Classifier models not found. Using dummy classifiers.")
            self.keypoint_classifier = None
            self.point_history_classifier = None
        
        # Load labels
        self.keypoint_labels = self._load_labels('model/keypoint_classifier/keypoint_classifier_label.csv')
        self.point_history_labels = self._load_labels('model/point_history_classifier/point_history_classifier_label.csv')
        
        # History tracking
        self.history_length = 16
        self.point_history = deque(maxlen=self.history_length)
        self.finger_gesture_history = deque(maxlen=self.history_length)
    
    def _load_labels(self, filepath: str) -> List[str]:
        """Load classifier labels from CSV file"""
        labels = []
        try:
            with open(filepath, 'r', encoding='utf-8-sig') as f:
                reader = csv.reader(f)
                labels = [row[0] for row in reader]
        except FileNotFoundError:
            logger.warning(f"Label file not found: {filepath}")
        return labels
    
    def process_frame(self, image: np.ndarray) -> Tuple[Optional[str], List, np.ndarray]:
        """Process a single frame and return gesture classification"""
        image_rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False
        results = self.hands.process(image_rgb)
        image_rgb.flags.writeable = True
        
        debug_image = copy.deepcopy(image)
        detected_command = None
        landmark_list = []
        
        if results.multi_hand_landmarks:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, 
                                                  results.multi_handedness):
                # Calculate landmarks
                landmark_list = self._calc_landmark_list(debug_image, hand_landmarks)
                brect = self._calc_bounding_rect(debug_image, hand_landmarks)
                
                # Preprocess landmarks
                processed_landmarks = self._preprocess_landmark(landmark_list)
                processed_point_history = self._preprocess_point_history(debug_image, self.point_history)
                
                # Classify hand sign
                if self.keypoint_classifier and processed_landmarks:
                    hand_sign_id = self.keypoint_classifier(processed_landmarks)
                    if isinstance(hand_sign_id, int) and 0 <= hand_sign_id < len(self.keypoint_labels):
                        detected_command = self.keypoint_labels[hand_sign_id]
                        self.point_history.append(landmark_list[8])  # Index fingertip
                    else:
                        self.point_history.append([0, 0])
                
                # Classify finger gesture
                finger_gesture_id = 0
                if (self.point_history_classifier and 
                    len(processed_point_history) == (self.history_length * 2)):
                    finger_gesture_id = self.point_history_classifier(processed_point_history)
                
                self.finger_gesture_history.append(finger_gesture_id)
                most_common_fg = Counter(self.finger_gesture_history).most_common(1)
                
                # Draw on debug image
                debug_image = self._draw_landmarks(debug_image, landmark_list)
                debug_image = self._draw_bounding_rect(debug_image, brect)
                debug_image = self._draw_info_text(
                    debug_image, brect, handedness, 
                    detected_command or "", 
                    self.point_history_labels[most_common_fg[0][0]] if most_common_fg and self.point_history_labels else ""
                )
        else:
            self.point_history.append([0, 0])
        
        debug_image = self._draw_point_history(debug_image, self.point_history)
        
        return detected_command, landmark_list, debug_image
    
    def _calc_landmark_list(self, image: np.ndarray, landmarks) -> List[List[int]]:
        """Calculate landmark coordinates"""
        image_height, image_width = image.shape[:2]
        landmark_points = []
        
        for landmark in landmarks.landmark:
            x = min(int(landmark.x * image_width), image_width - 1)
            y = min(int(landmark.y * image_height), image_height - 1)
            landmark_points.append([x, y])
        
        return landmark_points
    
    def _calc_bounding_rect(self, image: np.ndarray, landmarks) -> List[int]:
        """Calculate bounding rectangle for hand"""
        image_height, image_width = image.shape[:2]
        landmark_array = np.empty((0, 2), int)
        
        for landmark in landmarks.landmark:
            x = min(int(landmark.x * image_width), image_width - 1)
            y = min(int(landmark.y * image_height), image_height - 1)
            landmark_array = np.append(landmark_array, [[x, y]], axis=0)
        
        x, y, w, h = cv.boundingRect(landmark_array)
        return [x, y, x + w, y + h]
    
    def _preprocess_landmark(self, landmark_list: List[List[int]]) -> List[float]:
        """Preprocess landmarks for classification"""
        if not landmark_list:
            return []
        
        temp_landmarks = copy.deepcopy(landmark_list)
        
        # Convert to relative coordinates
        base_x, base_y = temp_landmarks[0]
        for i, point in enumerate(temp_landmarks):
            temp_landmarks[i][0] -= base_x
            temp_landmarks[i][1] -= base_y
        
        # Flatten and normalize
        flat_landmarks = list(itertools.chain.from_iterable(temp_landmarks))
        max_value = max(map(abs, flat_landmarks)) if flat_landmarks else 1
        
        return [n / max_value for n in flat_landmarks]
    
    def _preprocess_point_history(self, image: np.ndarray, point_history: deque) -> List[float]:
        """Preprocess point history for classification"""
        image_height, image_width = image.shape[:2]
        temp_history = copy.deepcopy(point_history)
        
        if not temp_history:
            return []
        
        # Convert to relative coordinates
        base_x, base_y = temp_history[0] if temp_history[0] != [0, 0] else [0, 0]
        for i, point in enumerate(temp_history):
            if point != [0, 0]:
                temp_history[i][0] = (point[0] - base_x) / image_width
                temp_history[i][1] = (point[1] - base_y) / image_height
        
        return list(itertools.chain.from_iterable(temp_history))
    
    def _draw_landmarks(self, image: np.ndarray, landmark_points: List[List[int]]) -> np.ndarray:
        """Draw hand landmarks on image"""
        if not landmark_points or len(landmark_points) < 21:
            return image
        
        # Hand connections (simplified)
        connections = [
            (0, 1), (1, 2), (2, 5), (5, 9), (9, 13), (13, 17), (17, 0),  # Palm
            (2, 3), (3, 4),  # Thumb
            (5, 6), (6, 7), (7, 8),  # Index
            (9, 10), (10, 11), (11, 12),  # Middle
            (13, 14), (14, 15), (15, 16),  # Ring
            (17, 18), (18, 19), (19, 20),  # Pinky
        ]
        
        # Draw connections
        for start_idx, end_idx in connections:
            if start_idx < len(landmark_points) and end_idx < len(landmark_points):
                start_point = tuple(landmark_points[start_idx])
                end_point = tuple(landmark_points[end_idx])
                cv.line(image, start_point, end_point, (255, 255, 255), 2)
        
        # Draw landmarks
        for i, point in enumerate(landmark_points):
            radius = 8 if i in [4, 8, 12, 16, 20] else 5  # Fingertips larger
            cv.circle(image, tuple(point), radius, (255, 255, 255), -1)
            cv.circle(image, tuple(point), radius, (0, 0, 0), 1)
        
        return image
    
    def _draw_bounding_rect(self, image: np.ndarray, brect: List[int]) -> np.ndarray:
        """Draw bounding rectangle"""
        cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[3]), (0, 255, 0), 2)
        return image
    
    def _draw_info_text(self, image: np.ndarray, brect: List[int], handedness, 
                       hand_sign_text: str, finger_gesture_text: str) -> np.ndarray:
        """Draw information text"""
        cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[1] - 22), (0, 0, 0), -1)
        
        info_text = handedness.classification[0].label
        if hand_sign_text:
            info_text += f': {hand_sign_text}'
        
        cv.putText(image, info_text, (brect[0] + 5, brect[1] - 4),
                   cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        return image
    
    def _draw_point_history(self, image: np.ndarray, point_history: deque) -> np.ndarray:
        """Draw point history trail"""
        for i, point in enumerate(point_history):
            if point[0] != 0 and point[1] != 0:
                cv.circle(image, tuple(point), 1 + int(i / 2), (152, 251, 152), 2)
        return image


class FPSCalculator:
    """Simple FPS calculator"""
    
    def __init__(self, buffer_len: int = 10):
        self.buffer_len = buffer_len
        self.timestamps = deque(maxlen=buffer_len)
    
    def update(self) -> float:
        """Update and return current FPS"""
        self.timestamps.append(time.time())
        if len(self.timestamps) < 2:
            return 0.0
        
        time_diff = self.timestamps[-1] - self.timestamps[0]
        return (len(self.timestamps) - 1) / time_diff if time_diff > 0 else 0.0


class DroneGestureController:
    """Main application class"""
    
    def __init__(self, args):
        self.args = args
        self.drone_controller = DroneController()
        self.gesture_recognizer = HandGestureRecognizer(
            min_detection_confidence=args.min_detection_confidence,
            min_tracking_confidence=args.min_tracking_confidence
        )
        self.fps_calculator = FPSCalculator()
        
        # Command cooldown to prevent spam
        self.last_command = None
        self.last_command_time = 0
        self.command_cooldown = 2.0  # seconds
        
        # Setup camera
        self.cap = cv.VideoCapture(args.device)
        self.cap.set(cv.CAP_PROP_FRAME_WIDTH, args.width)
        self.cap.set(cv.CAP_PROP_FRAME_HEIGHT, args.height)
    
    def run(self):
        """Main application loop"""
        logger.info("Starting drone gesture control application")
        
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    logger.error("Failed to read from camera")
                    break
                
                # Mirror the frame
                frame = cv.flip(frame, 1)
                
                # Process frame
                command, landmark_list, debug_image = self.gesture_recognizer.process_frame(frame)
                
                # Execute drone command with cooldown
                current_time = time.time()
                if command and (command != self.last_command or 
                               (current_time - self.last_command_time) > self.command_cooldown):
                    
                    if command == "Follow":
                        # Follow command can be executed continuously
                        self.drone_controller.execute_command(command, landmark_list)
                    else:
                        # Other commands have cooldown
                        self.drone_controller.execute_command(command, landmark_list)
                        self.last_command = command
                        self.last_command_time = current_time
                
                # Update FPS and draw info
                fps = self.fps_calculator.update()
                self._draw_info(debug_image, fps)
                
                # Display
                cv.imshow('Hand Gesture Drone Control', debug_image)
                
                # Check for exit
                key = cv.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:  # 'q' or ESC
                    break
                    
        except KeyboardInterrupt:
            logger.info("Application interrupted by user")
        except Exception as e:
            logger.error(f"Application error: {e}")
        finally:
            self.cleanup()
    
    def _draw_info(self, image: np.ndarray, fps: float):
        """Draw application info on image"""
        cv.putText(image, f"FPS: {fps:.1f}", (10, 30), 
                   cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        cv.putText(image, "Press 'q' or ESC to quit", (10, image.shape[0] - 20), 
                   cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
    def cleanup(self):
        """Cleanup resources"""
        self.cap.release()
        cv.destroyAllWindows()
        logger.info("Application cleanup completed")


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Hand Gesture Drone Control")
    
    parser.add_argument("--device", type=int, default=0, help="Camera device index")
    parser.add_argument("--width", type=int, default=960, help="Camera width")
    parser.add_argument("--height", type=int, default=540, help="Camera height")
    parser.add_argument("--min_detection_confidence", type=float, default=0.7,
                       help="Minimum detection confidence")
    parser.add_argument("--min_tracking_confidence", type=float, default=0.5,
                       help="Minimum tracking confidence")
    parser.add_argument("--connection", type=str, default='udp:127.0.0.1:14550',
                       help="MAVLink connection string")
    
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()
    app = DroneGestureController(args)
    app.run()
