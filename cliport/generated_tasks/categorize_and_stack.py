import numpy as np
from cliport.tasks.task import Task
from cliport.utils import utils
import pybullet as p

class CategorizeAndStack(Task):
    """Pick up and categorize the objects based on their type (e.g., cookies, mustard, raisins) and stack them into separate piles."""

    def __init__(self):
        super().__init__()
        self.max_steps = 20
        self.lang_template = "categorize and stack the {obj} in their designated areas"
        self.task_completed_desc = "done categorizing and stacking."
        self.additional_reset()

    def reset(self, env):
        super().reset(env)

        # Define the objects to be used in the task
        objects = [
            ('HOPE/Cookies.urdf', 'cookie', (0.1, 0.1, 0.2)),
            ('HOPE/Mustard.urdf', 'mustard', (0.05, 0.05, 0.15)),
            ('HOPE/Raisins.urdf', 'raisin', (0.05, 0.05, 0.05)),
            ('HOPE/MacaroniAndCheese.urdf', 'macaroni', (0.1, 0.1, 0.15)),
            ('HOPE/OrangeJuice.urdf', 'juice', (0.1, 0.1, 0.15))
        ]

        # Define the bounding boxes for the objects
        bounding_boxes = {
            'HOPE/MacaroniAndCheese.urdf': [np.array([0.40073031, 0.09382602, 0.02779107]), np.array([0.24480234, 0.27805897, 0.20423028])],
            'HOPE/OrangeJuice.urdf': [np.array([0.58768185, 0.18736881, -0.0091569]), np.array([0.4654693, 0.33479936, 0.05533345])],
            'HOPE/Raisins.urdf': [np.array([0.59408754, -0.23607832, -0.01133382]), np.array([0.49791071, -0.11149889, 0.02307193])],
            'HOPE/Mustard.urdf': [np.array([0.51985904, 0.0109054, 0.02402921]), np.array([0.30001166, 0.10895201, 0.19113784])],
            'HOPE/Cookies.urdf': [np.array([0.54285512, -0.26081692, 0.04158284]), np.array([0.27063629, -0.13746825, 0.21828878])],
            'HOPE/Cookies.urdf': [np.array([0.56807585, -0.41929029, 0.0419795]), np.array([0.30528927, -0.25100372, 0.24044817])],
            'HOPE/Raisins.urdf': [np.array([0.63189748, -0.35931477, -0.0063346]), np.array([0.53215266, -0.23431437, 0.02889575])]
        }

        # Define positions for the categorized piles
        pile_positions = {
            'cookie': (0.3, -0.3, 0.15),
            'mustard': (0.4, -0.3, 0.15),
            'raisin': (0.5, -0.3, 0.15),
            'macaroni': (0.3, 0.3, 0.15),
            'juice': (0.4, 0.3, 0.15)
        }

        # Add each object to the environment
        obj_ids = []
        for urdf, category, size in objects:
            if urdf in bounding_boxes:
                pose = (bounding_boxes[urdf][0], utils.eulerXYZ_to_quatXYZW((0, 0, 0)))
            else:
                pose = self.get_random_pose(env, size)
            obj_id = env.add_object(urdf, pose)
            obj_ids.append((obj_id, category))

        # Set up goals for each category of objects
        for category, target_pose in pile_positions.items():
            category_objs = [obj_id for obj_id, cat in obj_ids if cat == category]
            target_poses = [target_pose] * len(category_objs)
            matches = np.ones((len(category_objs), len(target_poses)))
            language_goal = self.lang_template.format(obj=category)
            
            self.add_goal(
                objs=category_objs, 
                matches=matches, 
                targ_poses=target_poses, 
                replace=False,
                rotations=True, 
                metric='pose', 
                params=None, 
                step_max_reward=1 / len(pile_positions), 
                language_goal=language_goal
            )