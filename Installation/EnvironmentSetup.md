# How to set up your Robot and Environment
This README provides a step-by-step guide on setting up your environment and connecting to the PenguinPi robot.

- [Setting up your device](#setting-up-your-device)
    - [Windows - Installing Anaconda](#installing-conda-and-requirements-on-windows)
    - [MacOS - Installing OrbStack](#installing-orbstack-on-macos)
    - [Linux Ubuntu - Installing Docker](#installing-docker-on-linux-ubuntu)
    - [Running the Container (Linux and Mac)](#running-the-container)
- [Optional (Using VSCode Editor)](#optional-using-vscode-editor)
- [Troubleshooting](#troubleshooting)

<div style="page-break-after: always"></div>

# Setting up your device

For Windows users, you will install all the required software via [Anaconda](https://www.anaconda.com/download). Note that if you are cloning the repository in Windows, **make sure that the repository is not cloned with OneDrive as this causes permission errors**. For Mac users, install [OrbStack](https://orbstack.dev/download) on your computer. For Linux user, just install `docker` according to your distro. The following sections will try to summarise the installation process for each OS. 

## Installing conda and required software on Windows

1. Install [Anaconda](https://www.anaconda.com/download). We recommend selecting "Add Anaconda3 to my PATH environment variable" as shown below. If you already have Anaconda installed and do not have it added to PATH but would like it to be then we have instrucitons to do so [below](#adding-anaconda-to-path).

    <img src="./pics/anaconda%20install%20settings.png" height="300" alt="Anaconda Install settings">

2. Open Anaconda Navigator

3. Select "Environments" form the bar on the left

4. Select the "Import" button at the bottom of the requirements list

5. Click on the folder icon next to the text box under local drive

6. Navigate to the directory that you cloned this repo into and select ```CondaEnv.yaml```

7. Change the name under "New environment name" to something more useful like "ECE4078_Lab"

8. Click "Import" and wait for the importing process to finish

### Using the Anaconda env

#### In the terminal

If you selected add to path in step 1 then in your terminal you can type the following where ```ENVNAME``` is the name you set for the environment in step 7

```
conda activate ENVNAME
```

You should see ```(ENVNAME)``` on the left of the input line of the terminal. This means you are in the env and all of the relevant packages should be avaliable to you. You can install new packages through the conda terminal or navigator as well as using pip. Please do not update packages that are already installed as you may create conflicts with the codebase.

Enter the following to exit the env

```
conda deactivate
```

Otherwise you can still use the provided "Anaconda Prompt" either by searching in windows or launching it from the Anaconda Navigator home page.

#### In VS Code

1. Open a .py file in vs code

2. Click on the python version number in the bottom right corner of the window

3. The environment your Anaconda environments should appear in the list as options which you can select

4. The version number in the bottom right corner should update to reflect the selection you made.

If you added Anaconda to path then you can also call conda activate in the vs code terminal to ues it there.

### Adding Anaconda to PATH
If you forgot to check the box during install, or already had Anaconda installed without adding it to PATH, then you can use the following instructions to add it to your systme PATH environment variable.

1. Search for ```Edit the system environment variables``` in the start menu or search bar.

2. Click the ```Environment Variables...``` button at the bottom the Sytem Properties window.

3. In the top box (labelled "User variables for *username*") select the ```Path``` variable and click the ```Edit``` button.

4. In the "Edit environment variable" window click the ```New``` button on the right and add the path of your anaconda installation. This is likely something like ```C:\Users\<USER>\anaconda3``` if you did a user level install or ```C:\ProgramData\Anaconda3``` if you did a system level install.

5. Repeat step 4 but this time at ```\scripts``` to the end of the file path e.g. ```C:\Users\<USER>\anaconda3\scripts```

6. Select ```OK``` on all of the windows that opened during this process.

7. Close any terminals that were open before making these changes so they can refresh their PATH list when opened again.

## Installing OrbStack on MacOS

**NOTE** If you already have experience with using Anaconda Navigator, and run into issues with dockerised containers, use Anaconda Navigator to create your new virtual environment.
1. Install Anaconda Navigator
2. Go to "Environments" tab on the left side of the app
3. Click on "Create" on the bottom left of the app
4. Name your environment something useful like "ECE4078_Lab"
5. Check "Python" and make sure the Python version selected is on 3.9.23 then click "Create"
6. Once your environment's created, click on your environment name, click the play button and "Open Terminal"
7. Change directory to the Installation folder of this repository and run `pip install -r requirements.txt`
8. And now you should have your virtual environment set up for the labs!

Otherwise, if you prefer to use a dockerised container on Orbstack, you may ignore the above instructions and continue from here...
Download and install [OrbStack](https://orbstack.dev/download). Note, choose the correct chip architecture (e.g. newer Macs have the Apple Silicon).

### Setting up Git Access
It is recommended that you clone **via SSH** so it's easier to set up version control within your dev container. [Generate a new SSH key](https://docs.github.com/en/authentication/connecting-to-github-with-ssh/generating-a-new-ssh-key-and-adding-it-to-the-ssh-agent). Follow the Mac, Github CLI steps, and stop once you reach the `Generating a new SSH key for a hardware security key` section. If you followed the steps correctly, you should be able to run `cat ~/.ssh/id_ed25519.pub` in your terminal, and copy-paste the output [here, when adding a new SSH key to your Github account](https://github.com/settings/keys)

### Setting up GUI Access

 Then:

1. Clone this repository to your local machine **via SSH**. 
2. Install the [XQuartz package here](https://www.xquartz.org/). Once installed, open XQuartz and go to `Settings > Security` and tick the option for `Allow connections from network clients`, then restart XQuartz.
3. Open up your terminal and run `defaults write org.xquartz.X11 enable_iglx -bool true`
4. Launch OrbStack, then go to terminal again and run `docker pull adedayoakinade/ece4078:noetic`
5. After connecting to the robot's hotspot, run the terminal command `./start_container.sh macos`. A warning may appear, about the image's platform not matching host platform. Ignore that error. Also, your terminal should have user `ece4078@...` now.
6. Now (again, but now **within your dev container**), follow the **Linux** steps and generate and add a new SSH key to your GitHub account.
7. To exit out of the container, type `exit` in the terminal. To enter back into the container, you simply run the `start_container` script again.

**NOTE:** If you are going to run on a specific PenguinPi for the first time, make sure you connect to the robot's hotspot, before you run `start_container`. It should add the robot IP to the access control list, to enable access for the docker container to parse the PenguinPi GUI onto XQuartz.

Now, you will notice the dockerised environment does not contain the ECE4078 repository. To copy the repository from your local device into the container, run `docker cp <source-path> ece4078_2025_lab:/home/ece4078/ECE4078_Lab/`. The source path is the directory to your repo folder.


## Installing Docker on Linux Ubuntu
Follow the [steps](https://docs.docker.com/engine/install/ubuntu/) on the official docker documentation to install the docker engine.

### Setting up GUI Access
In order to allow GUI access, for running the simulator, run the following command on a terminal on the host machine  to allow local root access to the X server:
```
xhost +local:root
```

<div style="page-break-after: always"></div>

---
# Running the Container
If you followed the MacOS and Linux instructions, then you should follow the instructions here to run your docker container. The docker image for the labs is available at: `adedayoakinade/ece4078:noetic`. It is based on Ubuntu 20.04 and ROS Noetic with Python 3.8.10. While it is setup for `sudo` access to not require password, the username and password is:

    username: ece4078
    password: 2025Lab

You can pull the image and run based on your preferences. However, a script `start_container.sh` is provided for starting a container from the docker image provided for the lab. The script pulls the image (if not previously pulled to your host machine) and starts a container. Simply open a terminal and run the script:
```
./start_container.sh <host_OS> <optional_container_name> <optional_image_name>
```

`<host_OS>` argument should be set based on your host. Options are `linux` or `macos`. The `<optional_container_name>` and `<optional_image_name>` arguments can be omitted as a default name, `ece4078_2025_lab` is chosen. If you want a specific name, you can set these arguments. This script returns the name of the running container (`<container_name>`). Take note of it as it can be useful to attach terminals from your host to the container where necessary. To do this:
```
docker attach <container_name>
```

<div style="page-break-after: always"></div>

# Optional (Using VSCode Editor)
You can use VSCode IDE for development in the contsiner. It also supports integrated terminal.
- Go to extensions and search for docker (`Dev Containers`). Install the extension.
- Go to `Remote Explorer` annd refresh the list of `Dev Containers`. Your running container should show up. If list is empty, it means the container is not running. Follow steps [above](#running-the-container) to run the container.
- Right click on the running container and `Attach in New Window`

You now get a full environment where you can drop files/folders from host to container and download files/foldersfrom the container to your host machine by right-clicking on the file.

<div style="page-break-after: always"></div>

---

# Troubleshooting
- [MacOS] If you have trouble with using Git within the docker container due to lack of permissions, try running `sudo chown -R ece4078:ece4078 /home/ece4078/ECE4078_Lab/` in your dev container terminal.
- [Windows] If you encounter the error "PermissionError: [WinError 5] Access is denied: 'pibot_dataset/'", please make sure you have not cloned the folder into OneDrive. If you have done the initial clone into OneDrive, you cannot just copy the folder over, you will need to reclone the repository into another folder on your drive, as the permissions for that folder will not be set correctly otherwise
- [Windows] If you get an error along the lines of "The term 'conda' is not recognized as a name of a cmdlet, function, script file, or executable program." then Anaconda has not been added to your system PATH environment variables. You can either [add Anaconda to your system PATH](#adding-anaconda-to-path) or use Anaconda Prompt from the system search or launched from Anaconda Navigator.


<div style="page-break-after: always"></div>

---
# Acknowledgement
- Part of the lab sessions are inspired by the [Robotic Vision Summer School](https://www.rvss.org.au/)
- Robot dev in collaboration with [QUT Centre for Robotics](https://github.com/qcr/PenguinPi-robot)
