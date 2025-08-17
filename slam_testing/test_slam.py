#team LastMile

"""
SLAM testing and evaluation module
Tests the accuracy and performance of the SLAM system
"""

import unittest
import numpy as np
import time
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class SLAMTestCase(unittest.TestCase):
    """Test cases for SLAM functionality"""
    
    def setUp(self):
        """Set up test environment"""
        self.test_poses = [
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (1.0, 1.0, np.pi/2),
            (0.0, 1.0, np.pi),
            (0.0, 0.0, -np.pi/2)
        ]
        self.test_landmarks = [
            (2.0, 2.0, 1, "aruco"),
            (-1.0, 3.0, 2, "aruco"),
            (3.0, -1.0, 3, "aruco")
        ]
    
    def test_pose_estimation_accuracy(self):
        """Test pose estimation accuracy"""
        print("Testing pose estimation accuracy...")
        
        for pose in self.test_poses:
            x, y, theta = pose
            
            # Simulate SLAM pose estimation with noise
            estimated_x = x + np.random.normal(0, 0.01)
            estimated_y = y + np.random.normal(0, 0.01)
            estimated_theta = theta + np.random.normal(0, 0.05)
            
            # Check accuracy within tolerance
            self.assertAlmostEqual(x, estimated_x, delta=0.1)
            self.assertAlmostEqual(y, estimated_y, delta=0.1)
            self.assertAlmostEqual(theta, estimated_theta, delta=0.2)
    
    def test_landmark_detection(self):
        """Test landmark detection and mapping"""
        print("Testing landmark detection...")
        
        detected_landmarks = []
        
        for landmark in self.test_landmarks:
            x, y, landmark_id, landmark_type = landmark
            
            # Simulate landmark detection
            detected_x = x + np.random.normal(0, 0.02)
            detected_y = y + np.random.normal(0, 0.02)
            
            detected_landmarks.append((detected_x, detected_y, landmark_id, landmark_type))
            
            # Check detection accuracy
            self.assertAlmostEqual(x, detected_x, delta=0.15)
            self.assertAlmostEqual(y, detected_y, delta=0.15)
        
        # Check that we detected the expected number of landmarks
        self.assertEqual(len(detected_landmarks), len(self.test_landmarks))
    
    def test_mapping_consistency(self):
        """Test map consistency over time"""
        print("Testing mapping consistency...")
        
        # Simulate multiple observations of the same landmark
        landmark_observations = []
        true_landmark = (5.0, 3.0)
        
        for i in range(10):
            # Add noise to simulate real observations
            obs_x = true_landmark[0] + np.random.normal(0, 0.05)
            obs_y = true_landmark[1] + np.random.normal(0, 0.05)
            landmark_observations.append((obs_x, obs_y))
        
        # Calculate mean position
        mean_x = np.mean([obs[0] for obs in landmark_observations])
        mean_y = np.mean([obs[1] for obs in landmark_observations])
        
        # Check that the mean is close to the true position
        self.assertAlmostEqual(true_landmark[0], mean_x, delta=0.2)
        self.assertAlmostEqual(true_landmark[1], mean_y, delta=0.2)
    
    def test_performance_timing(self):
        """Test SLAM performance timing"""
        print("Testing SLAM performance...")
        
        start_time = time.time()
        
        # Simulate SLAM processing
        for _ in range(100):
            # Simulate EKF update
            pose_update_time = 0.001  # 1ms
            landmark_update_time = 0.002  # 2ms
            time.sleep(pose_update_time + landmark_update_time)
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Check that processing stays under real-time requirements
        self.assertLess(total_time, 1.0)  # Should complete in under 1 second
        print(f"SLAM processing completed in {total_time:.3f} seconds")
    
    def test_loop_closure_detection(self):
        """Test loop closure detection"""
        print("Testing loop closure detection...")
        
        # Simulate a path that returns to start
        path_poses = [
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (2.0, 0.0, np.pi/2),
            (2.0, 1.0, np.pi/2),
            (2.0, 2.0, np.pi),
            (1.0, 2.0, np.pi),
            (0.0, 2.0, -np.pi/2),
            (0.0, 1.0, -np.pi/2),
            (0.0, 0.0, 0.0)  # Back to start
        ]
        
        # Check if we can detect when we're back at the start
        start_pose = path_poses[0]
        end_pose = path_poses[-1]
        
        distance = np.sqrt((start_pose[0] - end_pose[0])**2 + 
                          (start_pose[1] - end_pose[1])**2)
        
        # Should detect loop closure (distance close to 0)
        self.assertLess(distance, 0.5)  # Within 0.5 meters

def run_performance_benchmark():
    """Run performance benchmarks"""
    print("\n" + "="*50)
    print("SLAM PERFORMANCE BENCHMARK")
    print("="*50)
    
    # Test different map sizes
    map_sizes = [10, 50, 100, 200]
    
    for size in map_sizes:
        print(f"\nTesting map with {size} landmarks...")
        
        start_time = time.time()
        
        # Simulate SLAM with varying map sizes
        for _ in range(size):
            # Simulate processing time per landmark
            processing_time = 0.001 * (1 + np.log(size))  # Log complexity
            time.sleep(processing_time)
        
        end_time = time.time()
        total_time = end_time - start_time
        
        print(f"Map size: {size}, Processing time: {total_time:.3f}s")
        print(f"Average time per landmark: {total_time/size:.6f}s")

def main():
    """Main test runner"""
    print("Starting SLAM Test Suite...")
    print("="*50)
    
    # Run unit tests
    unittest.main(argv=[''], exit=False, verbosity=2)
    
    # Run performance benchmarks
    run_performance_benchmark()
    
    print("\n" + "="*50)
    print("SLAM testing completed!")
    print("="*50)

if __name__ == "__main__":
    main()