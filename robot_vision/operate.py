#team LastMile

"""
Robot operation module for autonomous navigation and control
Handles SLAM, path planning, and robot movement
"""

import time
import numpy as np
import pygame
import sys
import argparse
import json
import socket
import threading
from dataclasses import dataclass
from typing import List, Tuple, Optional

@dataclass
class RobotPose:
    x: float
    y: float
    theta: float

@dataclass
class Landmark:
    x: float
    y: float
    id: int
    type: str

class RobotController:
    def __init__(self, ip="localhost", port=8000):
        """Initialize robot controller"""
        self.ip = ip
        self.port = port
        self.pose = RobotPose(0.0, 0.0, 0.0)
        self.landmarks = []
        self.running = True
        self.slam_active = False
        
        # Initialize pygame for GUI
        pygame.init()
        self.screen = pygame.display.set_mode((800, 600))
        pygame.display.set_caption("Robot Control Interface")
        self.clock = pygame.time.Clock()
        
    def connect_robot(self):
        """Connect to physical robot"""
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.connect((self.ip, self.port))
            print(f"Connected to robot at {self.ip}:{self.port}")
            return True
        except Exception as e:
            print(f"Failed to connect to robot: {e}")
            return False
    
    def send_command(self, command: dict):
        """Send command to robot"""
        try:
            message = json.dumps(command)
            self.socket.send(message.encode())
        except Exception as e:
            print(f"Error sending command: {e}")
    
    def move_robot(self, linear_vel: float, angular_vel: float):
        """Send movement command to robot"""
        command = {
            "type": "move",
            "linear_velocity": linear_vel,
            "angular_velocity": angular_vel
        }
        self.send_command(command)
    
    def start_slam(self):
        """Start SLAM process"""
        self.slam_active = True
        print("SLAM started")
        # Initialize SLAM here
        
    def stop_slam(self):
        """Stop SLAM process"""
        self.slam_active = False
        print("SLAM stopped")
    
    def update_pose(self, x: float, y: float, theta: float):
        """Update robot pose"""
        self.pose.x = x
        self.pose.y = y
        self.pose.theta = theta
    
    def add_landmark(self, x: float, y: float, landmark_id: int, landmark_type: str):
        """Add detected landmark"""
        landmark = Landmark(x, y, landmark_id, landmark_type)
        self.landmarks.append(landmark)
    
    def draw_interface(self):
        """Draw the control interface"""
        self.screen.fill((0, 0, 0))  # Black background
        
        # Draw robot pose
        robot_x = int(self.pose.x * 10 + 400)  # Scale and center
        robot_y = int(self.pose.y * 10 + 300)
        pygame.draw.circle(self.screen, (0, 255, 0), (robot_x, robot_y), 10)
        
        # Draw landmarks
        for landmark in self.landmarks:
            lm_x = int(landmark.x * 10 + 400)
            lm_y = int(landmark.y * 10 + 300)
            pygame.draw.circle(self.screen, (255, 0, 0), (lm_x, lm_y), 5)
        
        # Draw status text
        font = pygame.font.Font(None, 36)
        status_text = f"SLAM: {'ON' if self.slam_active else 'OFF'}"
        text_surface = font.render(status_text, True, (255, 255, 255))
        self.screen.blit(text_surface, (10, 10))
        
        pose_text = f"Pose: ({self.pose.x:.2f}, {self.pose.y:.2f}, {self.pose.theta:.2f})"
        pose_surface = font.render(pose_text, True, (255, 255, 255))
        self.screen.blit(pose_surface, (10, 50))
        
        pygame.display.flip()
    
    def handle_events(self):
        """Handle pygame events"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.running = False
                elif event.key == pygame.K_RETURN:
                    if self.slam_active:
                        self.stop_slam()
                    else:
                        self.start_slam()
                elif event.key == pygame.K_r:
                    self.update_pose(0, 0, 0)  # Reset pose
                elif event.key == pygame.K_s:
                    self.save_map()
        
        # Handle continuous key presses for movement
        keys = pygame.key.get_pressed()
        linear_vel = 0
        angular_vel = 0
        
        if keys[pygame.K_UP] or keys[pygame.K_w]:
            linear_vel = 0.5
        elif keys[pygame.K_DOWN] or keys[pygame.K_s]:
            linear_vel = -0.5
        
        if keys[pygame.K_LEFT] or keys[pygame.K_a]:
            angular_vel = 0.5
        elif keys[pygame.K_RIGHT] or keys[pygame.K_d]:
            angular_vel = -0.5
        
        if linear_vel != 0 or angular_vel != 0:
            self.move_robot(linear_vel, angular_vel)
    
    def save_map(self):
        """Save current map data"""
        map_data = {
            "pose": {
                "x": self.pose.x,
                "y": self.pose.y,
                "theta": self.pose.theta
            },
            "landmarks": [
                {
                    "x": lm.x,
                    "y": lm.y,
                    "id": lm.id,
                    "type": lm.type
                } for lm in self.landmarks
            ]
        }
        
        with open("map_data.json", "w") as f:
            json.dump(map_data, f, indent=2)
        print("Map saved to map_data.json")
    
    def run(self):
        """Main control loop"""
        if self.ip != "localhost":
            if not self.connect_robot():
                print("Running in simulation mode")
        
        while self.running:
            self.handle_events()
            self.draw_interface()
            self.clock.tick(30)  # 30 FPS
        
        pygame.quit()
        if hasattr(self, 'socket'):
            self.socket.close()

def main():
    parser = argparse.ArgumentParser(description='Robot Operation Control')
    parser.add_argument('--ip', type=str, default='localhost', help='Robot IP address')
    parser.add_argument('--port', type=int, default=8000, help='Robot port')
    
    args = parser.parse_args()
    
    controller = RobotController(args.ip, args.port)
    controller.run()

if __name__ == "__main__":
    main()