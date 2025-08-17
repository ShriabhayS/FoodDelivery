#team LastMile

"""
Image detection module for autonomous robot vision system
Handles real-time object detection using YOLOv8
"""

import cv2
import numpy as np
from ultralytics import YOLO
import argparse
import pygame
import sys

class ImageDetector:
    def __init__(self, demo_mode=False):
        """Initialize the image detector"""
        self.demo_mode = demo_mode
        self.model = None
        self.load_model()
    
    def load_model(self):
        """Load YOLOv8 model"""
        try:
            self.model = YOLO('yolov8n.pt')
            print("YOLOv8 model loaded successfully")
        except Exception as e:
            print(f"Error loading model: {e}")
    
    def detect_objects(self, frame):
        """Detect objects in the given frame"""
        if self.model is None:
            return frame, []
        
        results = self.model(frame)
        detections = []
        
        for r in results:
            boxes = r.boxes
            if boxes is not None:
                for box in boxes:
                    # Extract detection data
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = box.conf[0].cpu().numpy()
                    cls = int(box.cls[0].cpu().numpy())
                    
                    if conf > 0.5:  # Confidence threshold
                        detections.append({
                            'bbox': (int(x1), int(y1), int(x2), int(y2)),
                            'confidence': conf,
                            'class': cls
                        })
                        
                        # Draw bounding box
                        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                        cv2.putText(frame, f'Class: {cls} ({conf:.2f})', 
                                   (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        return frame, detections
    
    def run_demo(self):
        """Run in demo mode using webcam"""
        cap = cv2.VideoCapture(0)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Detect objects
            frame, detections = self.detect_objects(frame)
            
            # Display frame
            cv2.imshow('Object Detection - Demo Mode', frame)
            
            # Handle key events
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC key
                break
            elif key == ord(' '):  # Space key
                print(f"Detected {len(detections)} objects")
        
        cap.release()
        cv2.destroyAllWindows()

def main():
    parser = argparse.ArgumentParser(description='Image Detection for Robot Vision')
    parser.add_argument('--demo_mode', action='store_true', help='Run in demo mode')
    parser.add_argument('--ip', type=str, help='Robot IP address')
    parser.add_argument('--port', type=int, default=8000, help='Robot port')
    
    args = parser.parse_args()
    
    detector = ImageDetector(demo_mode=args.demo_mode)
    
    if args.demo_mode:
        detector.run_demo()
    else:
        print("Robot mode not implemented yet")

if __name__ == "__main__":
    main()