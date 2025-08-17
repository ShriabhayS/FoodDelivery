# Import the YOLO class from the ultralytics library for object detection.
from ultralytics import YOLO

# teleoperate the robot and perform SLAM
# will be extended in following milestones for system integration

# basic python packages
import numpy as np
import cv2 
import os, sys
import time

# import utility functions
#sys.path.insert(0, "{}/util".format(os.getcwd()))
from util.pibot import PenguinPi # access the robot
import util.DatasetHandler as dh # save/load functions
import util.measure as measure # measurements
import pygame # python package for GUI
import shutil # python package for file operations

# import SLAM components
sys.path.insert(0, "{}/slam".format(os.getcwd()))
from slam.ekf import EKF
from slam.robot import Robot
import slam.aruco_detector as aruco


class Operate:
    def __init__(self, args):
        self.folder = 'pibot_dataset/'
        if not os.path.exists(self.folder):
            os.makedirs(self.folder)
        else:
            shutil.rmtree(self.folder)
            os.makedirs(self.folder)
        
        # initialise data parameters
        if args.play_data:
            self.pibot = dh.DatasetPlayer("record")
        elif hasattr(args, 'demo_mode') and args.demo_mode:
            # Demo mode - use dummy robot for testing
            self.pibot = None
        else:
            self.pibot = PenguinPi(args.ip, args.port)

        # initialise SLAM parameters
        self.ekf = self.init_ekf(args.calib_dir, args.ip)
        self.aruco_det = aruco.aruco_detector(
            self.ekf.robot, marker_length = 0.07) # size of the ARUCO markers
        
        # Initialize SLAM state
        self.ekf_on = False
        self.request_recover_robot = False

        if args.save_data:
            self.data = dh.DatasetWriter('record')
        else:
            self.data = None
        self.output = dh.OutputWriter('lab_output')
        self.command = {'motion':[0, 0], 
                        'inference': False,
                        'output': False,
                        'save_inference': False,
                        'save_image': False}
        self.quit = False
        self.pred_fname = ''
        self.request_recover_robot = False
        self.file_output = None
        self.ekf_on = False
        self.double_reset_comfirm = 0
        self.image_id = 0
        self.notification = 'Press ENTER to start SLAM'
        # a 5min timer
        self.count_down = 300
        self.start_time = time.time()
        self.control_clock = time.time()
        # initialise images
        self.img = np.zeros([240,320,3], dtype=np.uint8)
        self.aruco_img = np.zeros([240,320,3], dtype=np.uint8)
        self.bg = pygame.image.load('pics/gui_mask.jpg')
        
        # --- YOLO INTEGRATION START ---
        # NOTE: For brand-specific detection (like Coke and Red Bull), you need a custom-trained model.
        # For this example, we'll continue using the general yolov8n.pt model and will filter the results
        # for generic classes like 'bottle', 'can', 'cup', and 'mug' to demonstrate the process.
        print("Initializing YOLOv8 model...")
        self.yolo_model = YOLO("yolov8n.pt")
        self.yolo_results = None
        self.yolo_img = None
        
        # Define the new image paths you want to use for file-based detection.
        # This list has been updated with the new path for the mug image.
        self.file_paths = [
            "/Users/jasminebaldevraj/Desktop/robotics/FoodDelivery/Week05-06/Screenshots/coke-final.jpg", # Path for a can
            "/Users/jasminebaldevraj/Desktop/robotics/FoodDelivery/Week05-06/Screenshots/mug3.jpeg" # Path for a mug
        ]
        
        # We've changed the list of classes to focus only on 'mug' and 'can'.
        self.target_classes = ['mug', 'can']
        # --- YOLO INTEGRATION END ---

    # wheel control
    def control(self):    
        if self.pibot is None:
            # Demo mode - return dummy drive measurement
            lv, rv = 0, 0
            dt = time.time() - self.control_clock
            drive_meas = measure.Drive(lv, rv, dt)
        elif args.play_data:
            lv, rv = self.pibot.set_velocity()            
            if not self.data is None:
                self.data.write_keyboard(lv, rv)
            dt = time.time() - self.control_clock
            drive_meas = measure.Drive(lv, rv, dt)
        else:
            lv, rv = self.pibot.set_velocity(
                self.command['motion'])
            if not self.data is None:
                self.data.write_keyboard(lv, rv)
            dt = time.time() - self.control_clock
            # running in sim
            if args.ip == 'localhost':
                drive_meas = measure.Drive(lv, rv, dt)
            # running on physical robot (right wheel reversed)
            else:
                drive_meas = measure.Drive(lv, -rv, dt)
        self.control_clock = time.time()
        return drive_meas
        
    # camera control
    def take_pic(self):
        if self.pibot is None:
            # Demo mode - cycle through test images
            import cv2
            test_images = ['Screenshots/coke-final.jpg', 'Screenshots/redbull.webp', 'Screenshots/mug3.jpeg']
            
            # Create image cycling counter if it doesn't exist
            if not hasattr(self, 'image_cycle_counter'):
                self.image_cycle_counter = 0
                self.image_cycle_timer = time.time()
            
            # Change image every 3 seconds
            if time.time() - self.image_cycle_timer > 3.0:
                self.image_cycle_counter = (self.image_cycle_counter + 1) % len(test_images)
                self.image_cycle_timer = time.time()
                print(f"Switching to image: {test_images[self.image_cycle_counter]}")
            
            try:
                # Load the current image in cycle
                img_path = test_images[self.image_cycle_counter]
                if os.path.exists(img_path):
                    self.img = cv2.imread(img_path)
                    self.img = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)
                    self.img = cv2.resize(self.img, (320, 240))
                else:
                    # Create a dummy image if file not found
                    self.img = np.zeros([240,320,3], dtype=np.uint8)
            except:
                self.img = np.zeros([240,320,3], dtype=np.uint8)
        else:
            self.img = self.pibot.get_image()
            if not self.data is None:
                self.data.write_image(self.img)

    # --- YOLO INTEGRATION START ---
    # New method to perform YOLO object detection on the current camera image
    def detect_yolo_objects(self):
        # Always run inference in demo mode or when inference command is true
        if self.pibot is None or self.command['inference']:
            print("Performing YOLO inference...")
            # The `predict` method returns a list of result objects.
            # We use `stream=False` to get all results at once.
            results = self.yolo_model.predict(source=self.img, conf=0.25, stream=False)
            
            # Get the results for the first (and only) image
            if results:
                self.yolo_results = results[0]
                
                # Draw the bounding boxes and labels on a copy of the image
                annotated_image = self.yolo_results.plot()
                self.yolo_img = annotated_image
                
                # Print detected objects and highlight cans/bottles
                if self.yolo_results.boxes is not None:
                    for box in self.yolo_results.boxes:
                        class_id = int(box.cls[0])
                        class_name = self.yolo_results.names[class_id]
                        confidence = box.conf[0].item()
                        
                        # Highlight if it's a bottle or can-like object
                        if class_name in ['bottle', 'can', 'cup', 'stop sign']:  # stop sign might be misidentified can
                            print(f"*** DETECTED TARGET: {class_name} (confidence: {confidence:.2f}) ***")
                        else:
                            print(f"Detected: {class_name} (confidence: {confidence:.2f})")
                
                # Turn off the inference command until the next request (only if not in demo mode)
                if self.pibot is not None:
                    self.command['inference'] = False
            else:
                self.yolo_results = None
                self.yolo_img = self.img.copy()

    def detect_yolo_on_file(self, file_path):
        """
        Runs YOLO detection on an image from a local file path.
        """
        print(f"Loading and performing YOLO inference on file: {file_path}")
        try:
            # Load the image from the specified file path
            img = cv2.imread(file_path)
            if img is None:
                print(f"Error: Could not read image from {file_path}")
                return
            
            # Run YOLO prediction on the loaded image
            results = self.yolo_model.predict(source=img, conf=0.25, stream=False)
            
            # Filter the results to only include detections of interest
            filtered_results = []
            if results and len(results) > 0:
                for box in results[0].boxes:
                    class_id = int(box.cls[0])
                    class_name = results[0].names[class_id]
                    if class_name in self.target_classes:
                        filtered_results.append(box)
                
            # Create a new results object with only the filtered boxes
            if filtered_results:
                self.yolo_results = results[0]
                self.yolo_results.boxes = filtered_results
                # Plot the filtered results on the image
                self.yolo_img = self.yolo_results.plot()
                print("Inference on file completed successfully.")
            else:
                self.yolo_results = None
                self.yolo_img = img.copy()
                print("No detections found for the target classes.")

        except Exception as e:
            print(f"An error occurred during file-based detection: {e}")
            self.yolo_results = None
            self.yolo_img = None
    # --- YOLO INTEGRATION END ---

    # SLAM with ARUCO markers       
    def update_slam(self, drive_meas):
        if self.pibot is None:
            # Demo mode - create dummy ArUco image
            self.aruco_img = self.img.copy()
            lms = []  # No landmarks in demo mode
        else:
            lms, self.aruco_img = self.aruco_det.detect_marker_positions(self.img)
            
        if self.request_recover_robot:
            is_success = self.ekf.recover_from_pause(lms)
            if is_success:
                self.notification = 'Robot pose is successfuly recovered'
                self.ekf_on = True
            else:
                self.notification = 'Recover failed, need >2 landmarks!'
                self.ekf_on = False
            self.request_recover_robot = False
        elif self.ekf_on: # and not self.debug_flag:
            self.ekf.predict(drive_meas)
            self.ekf.add_landmarks(lms)
            self.ekf.update(lms)

    # save images taken by the camera
    def save_image(self):
        f_ = os.path.join(self.folder, f'img_{self.image_id}.png')
        if self.command['save_image']:
            image = self.pibot.get_image()
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(f_, image)
            self.image_id += 1
            self.command['save_image'] = False
            self.notification = f'{f_} is saved'

    # wheel and camera calibration for SLAM
    def init_ekf(self, datadir, ip):
        fileK = "{}intrinsic.txt".format(datadir)
        camera_matrix = np.loadtxt(fileK, delimiter=',')
        fileD = "{}distCoeffs.txt".format(datadir)
        dist_coeffs = np.loadtxt(fileD, delimiter=',')
        fileS = "{}scale.txt".format(datadir)
        scale = np.loadtxt(fileS, delimiter=',')
        if ip == 'localhost':
            scale /= 2
        fileB = "{}baseline.txt".format(datadir)  
        baseline = np.loadtxt(fileB, delimiter=',')
        robot = Robot(baseline, scale, camera_matrix, dist_coeffs)
        return EKF(robot)

    # save SLAM map
    def record_data(self):
        if self.command['output']:
            self.output.write_map(self.ekf)
            self.notification = 'Map is saved'
            self.command['output'] = False

    # paint the GUI            
    def draw(self, canvas):
        canvas.blit(self.bg, (0, 0))
        text_colour = (220, 220, 220)
        v_pad = 40
        h_pad = 20

        # paint SLAM outputs
        ekf_view = self.ekf.draw_slam_state(res=(320, 480+v_pad),
            not_pause = self.ekf_on)
        canvas.blit(ekf_view, (2*h_pad+320, v_pad))
        robot_view = cv2.resize(self.aruco_img, (320, 240))
        self.draw_pygame_window(canvas, robot_view, 
                                position=(h_pad, v_pad) )
        # canvas.blit(self.gui_mask, (0, 0))
        self.put_caption(canvas, caption='SLAM', position=(2*h_pad+320, v_pad))
        # M2
        self.put_caption(canvas, caption='Detector (M2)', position=(h_pad, 240+2*v_pad))
        # M3
        self.put_caption(canvas, caption='PiBot Cam', position=(h_pad, v_pad))
        try:
            notifiation = TEXT_FONT.render(self.notification, False, text_colour)
        except:
            notifiation = pygame.font.Font(None, 24).render(self.notification, False, text_colour)
        canvas.blit(notifiation, (h_pad+10, 596))
        time_remain = self.count_down - time.time() + self.start_time
        if time_remain > 0:
            time_remain = f'Count Down: {time_remain:03.0f}s'
        elif int(time_remain)%2 == 0:
            time_remain = "Time Is Up !!!"
        else:
            time_remain = ""
        try:
            count_down_surface = TEXT_FONT.render(time_remain, False, (50, 50, 50))
        except:
            count_down_surface = pygame.font.Font(None, 24).render(time_remain, False, (50, 50, 50))
        canvas.blit(count_down_surface, (2*h_pad+320+10, 536))
        
        # --- YOLO INTEGRATION START ---
        # Display the YOLO detection results on the GUI.
        if self.yolo_img is not None:
            # We need to convert the image from BGR to RGB for Pygame.
            yolo_view = cv2.resize(self.yolo_img, (320, 240))
            self.draw_pygame_window(canvas, yolo_view, position=(h_pad, 240+2*v_pad))
        
        # Display detection logs if results are available.
        if self.yolo_results:
            log_text = "Detections:"
            for box in self.yolo_results.boxes:
                class_id = int(box.cls[0])
                class_name = self.yolo_results.names[class_id]
                confidence = box.conf[0].item()
                log_text += f"\n- {class_name}: {confidence:.2f}"
            
            # Render the log text
            try:
                log_surface = TEXT_FONT.render(log_text, False, text_colour)
            except:
                log_surface = pygame.font.Font(None, 24).render(log_text, False, text_colour)
            # You might need to adjust the position to prevent overlap
            canvas.blit(log_surface, (h_pad + 320, v_pad))
        # --- YOLO INTEGRATION END ---
        
    def update_keyboard(self):
        """Handle keyboard input for GUI"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.quit = True
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.quit = True
                elif event.key == pygame.K_SPACE:
                    self.command['inference'] = True
                    
    def draw_pygame_window(self, canvas, cv2_img, position):
        """Convert OpenCV image to pygame surface and blit it"""
        cv2_img = np.rot90(cv2_img)
        cv2_img = np.flipud(cv2_img)
        cv2_img = pygame.surfarray.make_surface(cv2_img)
        canvas.blit(cv2_img, position)
        
    def put_caption(self, canvas, caption, position):
        """Put caption text on canvas"""
        try:
            caption_surface = TEXT_FONT.render(caption, False, (220, 220, 220))
            canvas.blit(caption_surface, position)
        except:
            pass  # Skip if font not available

# --- MAIN LOOP UPDATE START ---
def main(args):
    # Add splash screen like operate.py
    pygame.init()
    TITLE_FONT = pygame.font.Font('pics/8-BitMadness.ttf', 35)
    TEXT_FONT = pygame.font.Font('pics/8-BitMadness.ttf', 40)
    
    width, height = 700, 660
    canvas = pygame.display.set_mode((width, height))
    pygame.display.set_caption('ECE4078 2024 Lab')
    pygame.display.set_icon(pygame.image.load('pics/8bit/pibot1.png'))
    canvas.fill((0, 0, 0))
    splash = pygame.image.load('pics/loading.png')
    pibot_animate = [pygame.image.load('pics/8bit/pibot1.png'),
                     pygame.image.load('pics/8bit/pibot2.png'),
                     pygame.image.load('pics/8bit/pibot3.png'),
                    pygame.image.load('pics/8bit/pibot4.png'),
                     pygame.image.load('pics/8bit/pibot5.png')]
    pygame.display.update()

    start = False
    counter = 40
    while not start:
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                start = True
        canvas.blit(splash, (0, 0))
        x_ = min(counter, 600)
        if x_ < 600:
            canvas.blit(pibot_animate[counter%10//2], (x_, 565))
            pygame.display.update()
            counter += 2

    # This is the main loop where you can now integrate the YOLO detection.
    op = Operate(args)
    
    while start:
        op.take_pic()
        
        # Call the YOLO detection function here
        op.detect_yolo_objects()
        
        # update SLAM
        op.update_slam(op.control())
        # drawing
        op.draw(canvas)
        op.update_keyboard()
        pygame.display.update()
# --- MAIN LOOP UPDATE END ---

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser("Yolo Integration")
    parser.add_argument("--ip", type=str, default='localhost', help="Robot IP address")
    parser.add_argument("--port", type=int, default=8000, help="Robot port number")
    parser.add_argument("--calib_dir", type=str, default="calibration/param/", help="Path to calibration files")
    parser.add_argument("--save_data", action='store_true', help="Save data to file")
    parser.add_argument("--play_data", action='store_true', help="Play data from file")
    parser.add_argument("--demo_mode", action='store_true', help="Run in demo mode without robot connection")
    args = parser.parse_args()
    
    # Check if the main loop has been updated before running
    # This is a placeholder to show the update, remove in final code
    main(args)
