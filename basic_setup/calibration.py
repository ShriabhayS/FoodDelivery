#team LastMile

"""
Camera calibration module for robot vision system
Handles camera parameter estimation and distortion correction
"""

import cv2
import numpy as np
import glob
import pickle
import os
import argparse

class CameraCalibrator:
    def __init__(self):
        """Initialize camera calibrator"""
        self.calibration_data = None
        self.camera_matrix = None
        self.distortion_coeffs = None
        self.calibration_images = []
        
    def prepare_object_points(self, checkerboard_size=(9, 6)):
        """Prepare 3D points for checkerboard pattern"""
        # Prepare object points (0,0,0), (1,0,0), (2,0,0) ....,(8,5,0)
        objp = np.zeros((checkerboard_size[0] * checkerboard_size[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:checkerboard_size[0], 0:checkerboard_size[1]].T.reshape(-1, 2)
        return objp
    
    def collect_calibration_data(self, image_path_pattern, checkerboard_size=(9, 6)):
        """Collect calibration data from checkerboard images"""
        # Termination criteria
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        
        # Prepare object points
        objp = self.prepare_object_points(checkerboard_size)
        
        # Arrays to store object points and image points
        objpoints = []  # 3d points in real world space
        imgpoints = []  # 2d points in image plane
        
        # Get list of calibration images
        images = glob.glob(image_path_pattern)
        
        if not images:
            print(f"No images found matching pattern: {image_path_pattern}")
            return False
        
        print(f"Found {len(images)} calibration images")
        
        for fname in images:
            img = cv2.imread(fname)
            if img is None:
                continue
                
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Find the chess board corners
            ret, corners = cv2.findChessboardCorners(gray, checkerboard_size, None)
            
            # If found, add object points, image points (after refining them)
            if ret:
                objpoints.append(objp)
                
                # Refine corner positions
                corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                imgpoints.append(corners2)
                
                # Draw and display the corners (optional)
                img = cv2.drawChessboardCorners(img, checkerboard_size, corners2, ret)
                
                print(f"Processed: {os.path.basename(fname)}")
            else:
                print(f"Checkerboard not found in: {os.path.basename(fname)}")
        
        if len(objpoints) < 10:
            print("Warning: Less than 10 good calibration images found")
            return False
        
        # Perform camera calibration
        img_shape = gray.shape[::-1]
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_shape, None, None)
        
        if ret:
            self.camera_matrix = mtx
            self.distortion_coeffs = dist
            self.calibration_data = {
                'camera_matrix': mtx,
                'distortion_coefficients': dist,
                'rotation_vectors': rvecs,
                'translation_vectors': tvecs,
                'image_shape': img_shape
            }
            
            # Calculate reprojection error
            total_error = 0
            for i in range(len(objpoints)):
                imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
                error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
                total_error += error
            
            mean_error = total_error / len(objpoints)
            print(f"Calibration completed successfully!")
            print(f"Mean reprojection error: {mean_error:.4f} pixels")
            
            return True
        else:
            print("Camera calibration failed")
            return False
    
    def save_calibration(self, filename='camera_calibration.pkl'):
        """Save calibration data to file"""
        if self.calibration_data is None:
            print("No calibration data to save")
            return False
        
        with open(filename, 'wb') as f:
            pickle.dump(self.calibration_data, f)
        
        print(f"Calibration data saved to {filename}")
        return True
    
    def load_calibration(self, filename='camera_calibration.pkl'):
        """Load calibration data from file"""
        try:
            with open(filename, 'rb') as f:
                self.calibration_data = pickle.load(f)
            
            self.camera_matrix = self.calibration_data['camera_matrix']
            self.distortion_coeffs = self.calibration_data['distortion_coefficients']
            
            print(f"Calibration data loaded from {filename}")
            return True
        except FileNotFoundError:
            print(f"Calibration file {filename} not found")
            return False
        except Exception as e:
            print(f"Error loading calibration: {e}")
            return False
    
    def undistort_image(self, image):
        """Remove distortion from image using calibration data"""
        if self.camera_matrix is None or self.distortion_coeffs is None:
            print("No calibration data available")
            return image
        
        return cv2.undistort(image, self.camera_matrix, self.distortion_coeffs, None, self.camera_matrix)
    
    def get_optimal_camera_matrix(self, image_shape, alpha=1):
        """Get optimal camera matrix for undistortion"""
        if self.camera_matrix is None or self.distortion_coeffs is None:
            return None
        
        h, w = image_shape[:2]
        new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
            self.camera_matrix, self.distortion_coeffs, (w, h), alpha, (w, h)
        )
        
        return new_camera_matrix, roi
    
    def print_calibration_info(self):
        """Print calibration information"""
        if self.calibration_data is None:
            print("No calibration data available")
            return
        
        print("\n" + "="*50)
        print("CAMERA CALIBRATION RESULTS")
        print("="*50)
        
        print(f"Image shape: {self.calibration_data['image_shape']}")
        print(f"\nCamera Matrix:")
        print(self.camera_matrix)
        print(f"\nDistortion Coefficients:")
        print(self.distortion_coeffs.ravel())
        
        # Calculate focal lengths and principal point
        fx = self.camera_matrix[0, 0]
        fy = self.camera_matrix[1, 1]
        cx = self.camera_matrix[0, 2]
        cy = self.camera_matrix[1, 2]
        
        print(f"\nFocal lengths: fx={fx:.2f}, fy={fy:.2f}")
        print(f"Principal point: cx={cx:.2f}, cy={cy:.2f}")

def main():
    """Main calibration routine"""
    parser = argparse.ArgumentParser(description='Camera Calibration Tool')
    parser.add_argument('--images', type=str, default='calibration_images/*.jpg',
                       help='Path pattern for calibration images')
    parser.add_argument('--checkerboard', type=str, default='9,6',
                       help='Checkerboard size (width,height)')
    parser.add_argument('--save', type=str, default='camera_calibration.pkl',
                       help='Output calibration file')
    parser.add_argument('--load', type=str, help='Load existing calibration file')
    parser.add_argument('--test_image', type=str, help='Test undistortion on image')
    
    args = parser.parse_args()
    
    calibrator = CameraCalibrator()
    
    if args.load:
        # Load existing calibration
        calibrator.load_calibration(args.load)
        calibrator.print_calibration_info()
    else:
        # Perform new calibration
        checkerboard_size = tuple(map(int, args.checkerboard.split(',')))
        
        print(f"Starting camera calibration...")
        print(f"Looking for images: {args.images}")
        print(f"Checkerboard size: {checkerboard_size}")
        
        if calibrator.collect_calibration_data(args.images, checkerboard_size):
            calibrator.save_calibration(args.save)
            calibrator.print_calibration_info()
        else:
            print("Calibration failed")
            return
    
    # Test undistortion if requested
    if args.test_image:
        test_img = cv2.imread(args.test_image)
        if test_img is not None:
            undistorted = calibrator.undistort_image(test_img)
            
            # Show comparison
            comparison = np.hstack((test_img, undistorted))
            cv2.imshow('Original vs Undistorted', comparison)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            
            # Save undistorted image
            output_name = f"undistorted_{os.path.basename(args.test_image)}"
            cv2.imwrite(output_name, undistorted)
            print(f"Undistorted image saved as {output_name}")

if __name__ == "__main__":
    main()