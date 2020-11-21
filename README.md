# Grasp_Insert_Project
Vrep Grasp&Insert Code Base

Code Overall Idea:
1) To grasp a target object with as little post-grasp displacement as possible
2) Match the target object and the hole to see if the object is insertable into the hole
3) Insert

Key features:
1) Minimal Displacement Grasping:
	1) Visual affordance network (img input --> CNN --> heatmap to represent grasp primitive)
	2) Self-supervised labeling (grasp score = f(dispacement), detailed equation see 		   get_grasp_label_value in grasp_trainer.py)

2) Matching:
	1) Two binary img patch to represent the target object and hole (Value '1' is used to denote 		   the pixels belongs to the target object or the hole part).
	2) The mature algorithm for now is iterative binary search.

3) Insert:
	1) Because of the grasping displacement, hope to use SAC to compensate it.


Requirements:
1) numpy, Pytorch 1.5.1
2) Coppeliasim (Previous Name: V_rep)
3) matplotlib

To Start Simulation Experiment:
1) Run Vrep
2) Open scene: "simulation/simulation.ttt"
3) To train the minimal grasping policy, run main_grasp_training.py
   To train the SAC insertion policy, run main_insert_trask.py
