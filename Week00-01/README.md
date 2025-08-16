# Checkpoint 1: Teleoperating the Robot with your Keyboard

We will be using the PenguinPi robot for our lab project.

This first checkpoint will allow you to become more familiar with the PenguinPi robot and how to program it.

## Objective 1: Setting up your Environment
1. The [Installation Guide](../Installation/EnvironmentSetup.md) provides detailed instructions on setting up your environment and connecting to the robot for development.
2. Within your dev environment, navigate to this week's lab by typing the commands

``` cd ~/ECE4078_Lab_2025/Week00-01/```

or if you are making your own virtual environment rather than the provided dockerised container

``` cd path/to/your/lab/folder/Week00-01/```

## Objective 2: Implement Keyboard Teleoperation

You will implement keyboard teleoperations by editing [line 142 - 155 of operate.py](operate.py#L142).
  - **Hint:** check out how the stop control is implemented at [line 158](operate.py#L158) and the wheel control function at [line 57-74](operate.py#L57).
  - You will also need [the pics folder for painting the GUI](pics/) and [the util folder for utility scripts](util/), already contained within the repo.
  - To control the physical robot, you will need to run ```python3 operate.py``` (see [Setup guide for how to connect to the physical robot](../Installation/HowToConnect.md)).
  - Below is what the teleoperation GUI (and loading menu) looks like:

![GUI Menu](pics/Menu.png?raw=true "GUI Menu")
![Teleop GUI](pics/Teleop.png?raw=true "Teleop GUI")

**Note:** In order to run ```operate.py```, your terminal needs to be in the directory where [operate.py](operate.py) is (eg. ```~/ECE4078_Lab_2025/Week00-01/```)

**You don't have to use the provided scripts.** If you like, feel free to be creative and write your own scripts for teleoperating the robot with keyboard.

---
## Marking Instructions 
You will need to demonstrate your teleoperation implementation on the robot during the Week 2 lab session **as individuals**. If you have finished the implementation this week, feel free to ask the demonstrators to mark you off during this week's session. 

Please follow the [marking steps](#marking-steps) and submit your implementation to Moodle before asking to be marked.

- C1 live demo tasks by completing [operate.py](operate.py):
  - Drive forward
  - Drive backward
  - Turn left
  - Turn right

**Please familiarise yourselves with these marking steps to ensure the demonstrators can finish marking you in the allocated time**
- [Marking steps](#marking-steps)
- [Marking checklist](#marking-checklist)

You will have a **STRICT** 10min time limit to get marked. You may open up the marking checklist, which is a simplified version of the following steps to remind yourself of the marking procedures. 


### Marking steps
#### Step 1:
**Do this BEFORE your lab session**

Zip your whole Week00-01 folder (including the util and pics folder, your modified operate.py with the teleoperate codes, your README file) and submit it via the Moodle submission box (according to your lab session). This submission is due by the starting time of the lab session, which means you should **submit your script BEFORE you come to the Week 2 lab**. 

**Tip:** Other than a README file describing your implementation, you may also include a text file in the zip file with a list of commands to use, if you don't know all the commands off by heart.

**[VERIFY]** When submitting codes for marking, only submit a Week00-01 zip that includes the required scripts for running the teleoperation (operate.py, util/, pics/) on Moodle. Then during marking, move the the downloaded and unzipped submission into your running container `~/LiveDemo` directory, cd into the directory, and run the live demo

Please check that you have clicked the submit button on Moodle (may need to scroll down a bit) when submitting your codes, instead of saving it as a draft.

#### Step 2: 
**Do this BEFORE the demonstrator come to mark you**

1. Close all the windows in your dev environment

2. Log in Moodle and navigate to the C1 submission box, so that you are ready to download your submitted code when it is time for your live demo marking


#### Step 3:
**During marking**
1. When the demonstrator start to mark you, download your submitted zip file from Moodle and unzip its content to the `~/LiveDemo` folder

2. Open another terminal, or new tab in the existing terminal, navigate to the "Week00-01" folder which contains the operate.py script

3. Connect to the physical robot and run the script with ```python3 operate.py``` (change the ip address if you need to)

4. Demonstrate the teleoperation and good luck!
---

### Marking checklist
**BEFORE the lab session**
- [ ] Submit your code to Moodle

**BEFORE the marking**
- [ ] Close all programs and folders
- [ ] Log in Moodle and navigate to the submission box
- [ ] Open an empty folder named "LiveDemo" (located at ```~/LiveDemo/```)

**During the marking**
- [ ] Demonstrator will ask you to download your submission and unzip it to "LiveDemo"
- [ ] If working with a dockerised or virtual environment, don't forget to transfer the "LiveDemo" folder into that environment
- [ ] Connect to the robot and run the operate.py script
- [ ] Demonstrate the teleoperation and good luck!

