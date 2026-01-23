---
layout: default
title: Proposal
---

## Summary of Project
Our project is to train an autonomous drone to fly in simulation through a forest to a particular destination without colliding with obstacles using Reinforcement Learning.
Our quadKWPter will recognize its surroundings using a camera to detect nearby obstacles (randomly placed trees). It will also know its current position, velocity, roll, pitch, yaw, angular velocity, and the position of its destination. From this information, the drone will accelerate, decelerate, or turn as necessary to avoid obstacles and continue making progress towards its destination.
We are hoping to bring this project to life one day with a physical drone and fly it through the trees in Aldrich Park. 


## Project Goals
* Minimum goal: Our minimum goal is to have our drone fly from a starting location A to a specified location B in a simulation. 
* Realistic goal: Our realistic goal is to have our drone fly through the forest from location A to location B without colliding into any trees along the way. 
* Moonshot goal: Our moonshot goal is to have our drone fly through the forest from location A to location B without colliding into any trees while also dealing with wind.


## AI/ML Algorithms
We will use PPO (and maybe compare it with DDPG) with rewards based on distance from target with penalties for flipping over, maybe using a depth camera data to identify obstacles and introducing an increasing penalty for getting too close to a tree.


## Evaluation Plan
To quantitatively evaluate the performance of our drone, we will run simulated experiments in which we have the drone navigate through randomly generated forest environments and we can measure metrics such as collision/success rate, distance travelled before collision, and time taken to go from the starting to finish point. The baseline that we can use to compare our RL performance to is to have a drone fly at a set velocity from point A to point B, which will perform poorly in forest settings (or it may not!), and we estimate that our RL approach can improve the baseline performance by a significant amount as the baseline approach’s success and failure rates will essentially be random.
For qualitative analysis, we can use toy examples such as getting our drone to fly in a straight line in an empty environment and maneuvering around a single tree to check the sanity and performance of our navigation and obstacle detection algorithm. We can visualize the trajectory and flight pattern of our drone along with maps of where the trees are as well as plot the training curves to verify that our RL policy is learning to meaningfully avoid the trees and understand the reasoning behind our drone’s behavior. A successful result will include smooth navigation around obstacles and consistent performance even in dense forest environments. 


