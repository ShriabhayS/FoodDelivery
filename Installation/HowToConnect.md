# How to set up your Robot
This README provides a step-by-step guide on what to do to connect to the PenguinPi robot after setting up your environment. This assumes you have already setup your environment according to your OS as specified in EnvironmentSetup.md

- [Inside the kit](#inside-the-kit)
- [Powering on and connecting to the robot](#powering-on-and-connecting-to-the-robot)
- [Running code on the robot](#running-code-on-the-robot)
- [Connecting to the robot and the internet](#connecting-to-the-robot-and-the-internet)
- [Disconnecting and shutting down the robot](#disconnecting-and-shutting-down-the-robot)
- [Troubleshooting](#troubleshooting)


---

## Inside the kit

The PenguinPi robot kit includes:
* one PenguinPi robot: details, see figure below
* one Wifi Dongle: your PenguinPi may not be able to generate a wifi hotspot, so it can plugged in the robot's usb port for generating wifi hotspot for a computer to connect to the robot. However, most robots don't need it so you may like to use the Wifi dongle for your laptop (see below)
* one USB cable: an extender of any usb ports on the robot for dev and debugging
* one charging cable: when fully charged, the light on the charger will change from red to green

The robot has an ID number printed on the carrying case and on the robot for logging ownership and repairs. Please don't mismatch the robot and its case.

![PenguinPi Robot Kit](../Images/PenguinPiKitAnno.png?raw=true "PenguinPi Robot kit")

Here is an overview of the robot's parts. **The camera is the front side of the robot** (the triangle part of the frame is the tail).

![PenguinPi Robot](../Images/PenguinPi.png?raw=true "PenguinPi Robot")

## Powering on and connecting to the robot
First, switch the robot on with the side switch

Wait for it to boot - this takes about 1 min. When IP addresses appears on the OLED screen, e.g., ```192.168.50.1```,  the booting is finished.

Connect your PC's wifi to the PenguinPi's hotspot penguinpi:xx:xx:xx (look for it in your wifi list). The hotspot password is ```PenguinPi```. You can identify which wifi belongs to your robot by the WMAC shown on the robot's screen (the last 6 digits/letters shown would match the hotspot's last 6 digits/letters).

You can test your connection by opening a web browser and entering the address ```192.168.50.1:8080``` ("192.168.50.1" is your robot's IP, "8080" is the port). On this webpage you will be able to test the motors and camera, as well as see a range of diagnostic information relating to your robot. **Please let us know if your robot's motors and/or camera aren't working so we can arrange for replace and repair.**

![Web browser view of the physical robot](../Images/WebRobot.png?raw=true "Web browser view of the physical robot")

If you want, you can also connect to the robot using SSH instead (the SSH password is ```PenguinPi```), replace the IP address if needed. Usually, you only SSH into the robot so you can shut the robot down safely, but the option to utilise the robot via SSH is available.

```
ssh -X pi@192.168.50.1
```

You can upload files to the robot using the ```scp``` command:
```
scp -r ./LOCALDIR pi@192.168.50.1:~/REMOTEDIR
```

## Connecting to the robot and the internet
You can use the WiFi dongles in the robot kits to add a second adaptor to your computer. This should allow you to connect to both the robot and an internet connected network at the same time. Just note that these adaptors are typically slower than what comes installed in most laptops these days so one of your connections may be a bit sluggish.

This method will connect the robot itself to the internet. You should not need to connect the robot to the internet to complete the tasks over the semester. If for some reason you do believe that you have a case that requires it then speak to a TA or make a post on the forums and we will help you sort it out.

## Running code on the robot
To run your code on the robot, you need to find the IP address of your robot. This is shown on its OLED screen.

You can also connect an external screen, keyboard, and mouse to the robot, then switch it on and use it as a Raspberry Pi. Inside the Raspberry Pi interface, you can install python packages onboard of the robot by running pip in the terminal, e.g., ```python3 -m pip install pynput```. You can also install packages inside the ssh session if your PenguinPi has internet connection (you can set the internet connection up in the Raspberry Pi interface).

## Disconnecting and shutting down the robot
When you are done with the robot, SSH into the robot, and inside the SSH session run ```sudo halt & logout``` to shut down the robot safely. Once the screen says 'TURNRaspberryPi OFF SAFELY' you can toggle the power switch. Avoid toggling the switch directly to "hard" shut down.

---
# Troubleshooting

- Please be gentle with the robot as they can be quite fragile
- Some of the robots may have a faulty camera (you can check this via the camera stream window on the web browser view). Please let us know if you have a robot with a faulty camera so we can arrange for replacement and repair.
- Some of the robots may have a faulty battery (if the charger LED blinks red when you are charging the robot and plug-unplug several times doesn't fix this). Please let us know if you have a robot with a faulty battery so we can arrange for replacement and repair.
- Some Windows 11 users may run into an "Execution Policy Settings" error, with an error message similar to "cannot be loaded because the execution of scripts is disabled on this system", when activating their Python venv. This can be fixed by typing ```Set-ExecutionPolicy Unrestricted -Scope Process``` in a terminal to change your Execution Policy Settings
- Some robots don't show the web browser preview due to an internal server error. You can ignore this error and check if you are connected to the robot by either ssh into it or by running operate.py

---
# Acknowledgement
- Part of the lab sessions are inspired by the [Robotic Vision Summer School](https://www.rvss.org.au/)
- Robot dev in collaboration with [QUT Centre for Robotics](https://github.com/qcr/PenguinPi-robot)
