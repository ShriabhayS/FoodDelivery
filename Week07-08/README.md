# Milestone 3: Navigation and Planning
- [Introduction](#introduction)
- [Waypoint navigation (Week 8)](#waypoint-navigation-week-8)
- Path planning
	- [with a known map (Week 9)](#path-planning-with-a-known-map-week-9)
	- [with a partially known map (Week 9)](#path-planning-with-a-partially-known-map-week-9)
- Evaluation and marking: see [M3 marking instructions](M3_marking.md)

**Please note that all skeleton codes provided are **for references only** and are intended to get you started. They are not guaranteed to be bug-free either. To get better performance please make changes or write your own codes and test cases. As long as your codes perform the task and generate estimation maps that can be marked according to the evaluation schemes and scripts you are free to modify the skeleton codes.**

---

## Introduction
In M3 you will implement the grocery shopping module for the robot. PenguinPi will be given a shopping list and it will need to navigate to the listed fruits&vegs in the supermarket. 

In the arena, there will be ArUco markers and fruits&vegs. The shopping list will contain a subset of the fruits&vegs present in the arena. The remaining fruits&vegs are considered obstacles. **The targets will be of unique types and there will be only one of each target fruit&vegs in the arena.** There may be more than one of a type of fruit/veg as obstacles. 
For example, in an arena containing 1 lemon, 1 lime, 1 garlic, 1 pumpkin, 1 tomato, 1 capsicum, 2 potatoes, 2 oranges, your robot's given shopping list may be "garlic, lemon, lime, tomato, pumpkin", with the capsicum, potatoes, and oranges as the obstacle. As there are duplicates of oranges and potatoes, the shopping list won't contain potatoes or oranges.

Since M3 focuses on testing your navigation and planning algorithms, we provide you with the groundtruth map of the arena. The repo contains map of a practice arena, slightly different marking arenas will be set up for M3 live demo marking and their true maps will be provided to you at the start of each M3 marking lab. You can either perform M3 with this fully known map, or with a partially known map if you are up for a challenge.

More details to be released closer to week 7.
