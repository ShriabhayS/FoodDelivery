#team LastMile

"""
Installation and setup script for the robot vision system
Handles environment setup and dependency installation
"""

import os
import sys
import subprocess
import platform
import argparse
from pathlib import Path

class SystemSetup:
    def __init__(self):
        """Initialize system setup"""
        self.platform = platform.system().lower()
        self.python_version = sys.version_info
        self.project_root = Path(__file__).parent.parent
        
    def check_python_version(self):
        """Check if Python version is compatible"""
        print("Checking Python version...")
        
        if self.python_version < (3, 8):
            print(f"❌ Python {self.python_version.major}.{self.python_version.minor} detected")
            print("❌ Python 3.8 or higher is required")
            return False
        
        print(f"✅ Python {self.python_version.major}.{self.python_version.minor}.{self.python_version.micro} is compatible")
        return True
    
    def check_conda(self):
        """Check if Conda is available"""
        try:
            result = subprocess.run(['conda', '--version'], capture_output=True, text=True)
            if result.returncode == 0:
                print(f"✅ Conda found: {result.stdout.strip()}")
                return True
        except FileNotFoundError:
            pass
        
        print("❌ Conda not found. Please install Anaconda or Miniconda")
        return False
    
    def create_conda_environment(self):
        """Create conda environment from yaml file"""
        env_file = self.project_root / "Installation" / "CondaEnv.yaml"
        
        if not env_file.exists():
            print("Creating CondaEnv.yaml...")
            self.create_conda_yaml()
        
        print("Creating conda environment...")
        try:
            subprocess.run(['conda', 'env', 'create', '-f', str(env_file)], check=True)
            print("✅ Conda environment 'robot_env' created successfully")
            return True
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to create conda environment: {e}")
            return False
    
    def create_conda_yaml(self):
        """Create conda environment yaml file"""
        conda_yaml_content = """name: robot_env
channels:
  - conda-forge
  - defaults
dependencies:
  - python=3.9
  - pip
  - numpy
  - opencv
  - scipy
  - matplotlib
  - jupyter
  - pip:
    - ultralytics
    - pygame
    - opencv-python
    - pillow
    - torch
    - torchvision
    - torchaudio
"""
        
        env_file = self.project_root / "Installation" / "CondaEnv.yaml"
        env_file.parent.mkdir(exist_ok=True)
        
        with open(env_file, 'w') as f:
            f.write(conda_yaml_content)
        
        print(f"✅ Created {env_file}")
    
    def create_requirements_txt(self):
        """Create requirements.txt file"""
        requirements_content = """# Core dependencies
numpy>=1.21.0
opencv-python>=4.5.0
scipy>=1.7.0
matplotlib>=3.4.0
pillow>=8.3.0

# Deep learning
torch>=1.9.0
torchvision>=0.10.0
torchaudio>=0.9.0
ultralytics>=8.0.0

# GUI and visualization
pygame>=2.0.0
jupyter>=1.0.0

# Robot communication
pyserial>=3.5
requests>=2.25.0

# Data processing
pandas>=1.3.0
scikit-learn>=1.0.0

# Testing
pytest>=6.0.0
unittest-xml-reporting>=3.0.0
"""
        
        req_file = self.project_root / "Installation" / "requirements.txt"
        req_file.parent.mkdir(exist_ok=True)
        
        with open(req_file, 'w') as f:
            f.write(requirements_content)
        
        print(f"✅ Created {req_file}")
    
    def install_pip_requirements(self):
        """Install pip requirements"""
        req_file = self.project_root / "Installation" / "requirements.txt"
        
        if not req_file.exists():
            print("Creating requirements.txt...")
            self.create_requirements_txt()
        
        print("Installing pip requirements...")
        try:
            subprocess.run([sys.executable, '-m', 'pip', 'install', '-r', str(req_file)], check=True)
            print("✅ Pip requirements installed successfully")
            return True
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install pip requirements: {e}")
            return False
    
    def setup_project_structure(self):
        """Create project directory structure"""
        print("Setting up project structure...")
        
        directories = [
            "robot_vision",
            "robot_vision/calibration",
            "robot_vision/calibration/param",
            "slam_testing",
            "basic_setup",
            "Installation",
            "image_to_map_generator",
            "data",
            "data/maps",
            "data/images",
            "data/calibration_images",
            "logs"
        ]
        
        for directory in directories:
            dir_path = self.project_root / directory
            dir_path.mkdir(exist_ok=True, parents=True)
            print(f"✅ Created directory: {directory}")
        
        return True
    
    def create_config_files(self):
        """Create configuration files"""
        print("Creating configuration files...")
        
        # Create config.yaml for image_to_map_generator
        config_content = """# Map generation configuration
map_settings:
  resolution: 0.05  # meters per pixel
  origin: [-10.0, -10.0, 0.0]  # map origin
  width: 400  # map width in pixels
  height: 400  # map height in pixels

detection_settings:
  confidence_threshold: 0.5
  nms_threshold: 0.4
  target_objects:
    - "bottle"
    - "cup"
    - "can"

slam_settings:
  max_landmarks: 100
  landmark_timeout: 30.0  # seconds
  pose_uncertainty_threshold: 0.1
"""
        
        config_file = self.project_root / "image_to_map_generator" / "config.yaml"
        config_file.parent.mkdir(exist_ok=True)
        
        with open(config_file, 'w') as f:
            f.write(config_content)
        
        print(f"✅ Created {config_file}")
        
        return True
    
    def install_system_dependencies(self):
        """Install system-level dependencies"""
        print("Checking system dependencies...")
        
        if self.platform == "linux":
            print("Linux detected - checking for required packages...")
            # Check for common Linux dependencies
            dependencies = ["libopencv-dev", "python3-dev", "build-essential"]
            print(f"Please ensure these packages are installed: {', '.join(dependencies)}")
            print("Run: sudo apt-get install " + " ".join(dependencies))
            
        elif self.platform == "darwin":  # macOS
            print("macOS detected - checking for Homebrew...")
            try:
                subprocess.run(['brew', '--version'], capture_output=True, check=True)
                print("✅ Homebrew found")
            except (FileNotFoundError, subprocess.CalledProcessError):
                print("❌ Homebrew not found. Please install from https://brew.sh/")
                return False
                
        elif self.platform == "windows":
            print("Windows detected - please ensure Visual C++ Build Tools are installed")
        
        return True
    
    def run_tests(self):
        """Run basic tests to verify installation"""
        print("Running installation tests...")
        
        test_imports = [
            "numpy",
            "cv2",
            "pygame",
            "ultralytics",
            "torch",
            "matplotlib"
        ]
        
        failed_imports = []
        
        for module in test_imports:
            try:
                __import__(module)
                print(f"✅ {module} imported successfully")
            except ImportError as e:
                print(f"❌ Failed to import {module}: {e}")
                failed_imports.append(module)
        
        if failed_imports:
            print(f"\n❌ Installation incomplete. Failed imports: {failed_imports}")
            return False
        
        print("\n✅ All modules imported successfully!")
        return True
    
    def print_completion_message(self):
        """Print setup completion message"""
        print("\n" + "="*60)
        print("🎉 SETUP COMPLETED SUCCESSFULLY!")
        print("="*60)
        print("\nNext steps:")
        print("1. Activate the conda environment:")
        print("   conda activate robot_env")
        print("\n2. Test the installation:")
        print("   cd robot_vision")
        print("   python imagedetect.py --demo_mode")
        print("\n3. For robot operation:")
        print("   python operate.py --ip <robot_ip>")
        print("\n4. For SLAM testing:")
        print("   cd slam_testing")
        print("   python test_slam.py")
        print("\n" + "="*60)

def main():
    """Main setup routine"""
    parser = argparse.ArgumentParser(description='Robot Vision System Setup')
    parser.add_argument('--skip-conda', action='store_true', help='Skip conda environment creation')
    parser.add_argument('--pip-only', action='store_true', help='Use pip only (no conda)')
    parser.add_argument('--test-only', action='store_true', help='Run tests only')
    
    args = parser.parse_args()
    
    setup = SystemSetup()
    
    print("🤖 Robot Vision System Setup")
    print("="*40)
    
    if args.test_only:
        setup.run_tests()
        return
    
    # Check basic requirements
    if not setup.check_python_version():
        sys.exit(1)
    
    if not setup.install_system_dependencies():
        print("⚠️  System dependency check failed, but continuing...")
    
    # Setup project structure
    setup.setup_project_structure()
    setup.create_config_files()
    
    # Handle environment setup
    if args.pip_only:
        setup.install_pip_requirements()
    elif not args.skip_conda:
        if setup.check_conda():
            setup.create_conda_environment()
        else:
            print("Falling back to pip installation...")
            setup.install_pip_requirements()
    
    # Run tests
    if setup.run_tests():
        setup.print_completion_message()
    else:
        print("\n❌ Setup completed with errors. Please check the failed imports.")

if __name__ == "__main__":
    main()