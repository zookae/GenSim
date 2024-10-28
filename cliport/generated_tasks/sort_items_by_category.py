import numpy as np
from cliport.tasks.task import Task
from cliport.utils import utils
import pybullet as p

class SortItemsByCategory(Task):
    """Pick up and sort the items into categories: place all food items on the left side of the table and non-food items on the right side."""

    def __init__(self):
        super().__init__()
        self.max_steps = 20  # Increased steps to give the agent more time to complete the task
        self.lang_template = "place the {item} on the {side} side of the table"
        self.task_completed_desc = "done sorting items by category."
        self.additional_reset()

    def reset(self, env):
        super().reset(env)

        # Define items and their categories with their bounding boxes
        items = [
            ('HOPE/GranolaBars.urdf', 'food', [np.array([0.48315348, 0.23598798, 0.00255921]), np.array([0.21079647, 0.51363411, 0.14675479])]),
            ('HOPE/Raisins.urdf', 'food', [np.array([ 0.51827382, -0.29651294, -0.01768806]), np.array([ 0.40659092, -0.17211218,  0.016381  ])]),
            ('HOPE/Corn.urdf', 'food', [np.array([ 0.65875019,  0.05674204, -0.00578587]), np.array([0.58252898, 0.13331217, 0.06450037])]),
            ('HOPE/Cherries.urdf', 'food', [np.array([ 0.4722811 , -0.43671704, -0.01075628]), np.array([ 0.36322675, -0.36328296,  0.05253737])]),
            ('block/block.urdf', 'non-food', [np.array([ 0.61892649, -0.08313771, -0.01618664]), np.array([ 0.54376944, -0.02154077,  0.03185457])]),
            ('HOPE/Mustard.urdf', 'non-food', [np.array([ 0.63181469,  0.16379597, -0.02225223]), np.array([0.53549881, 0.33232726, 0.02886849])])
        ]

        # Define target zones
        left_zone_size = (0.25, 0.5, 0)
        right_zone_size = (0.25, 0.5, 0)
        left_zone_pose = ((0.25, 0, 0.01), (0, 0, 0, 1))
        right_zone_pose = ((0.75, 0, 0.01), (0, 0, 0, 1))

        # Color zones for visual aid
        env.add_object('zone/zone.urdf', left_zone_pose, 'fixed')
        env.add_object('zone/zone.urdf', right_zone_pose, 'fixed')
        p.changeVisualShape(env.obj_ids['fixed'][-2], -1, rgbaColor=[0, 1, 0, 0.5])  # Green for food
        p.changeVisualShape(env.obj_ids['fixed'][-1], -1, rgbaColor=[1, 0, 0, 0.5])  # Red for non-food

        # Add items and set goals
        food_items = []
        non_food_items = []
        for urdf, category, bbox in items:
            size = (bbox[1][0] - bbox[0][0], bbox[1][1] - bbox[0][1], bbox[1][2] - bbox[0][2])
            pos = ((bbox[0][0] + bbox[1][0]) / 2, (bbox[0][1] + bbox[1][1]) / 2, (bbox[0][2] + bbox[1][2]) / 2)
            pose = (pos, (0, 0, 0, 1))
            obj_id = env.add_object(urdf, pose)
            if category == 'food':
                food_items.append(obj_id)
            else:
                non_food_items.append(obj_id)

        # Ensure that each goal has a higher reward to make the task more feasible
        goal_reward = 1 / (len(food_items) + len(non_food_items))

        # Goal: Place each food item on the left side of the table
        for i, obj_id in enumerate(food_items):
            self.add_goal(objs=[obj_id], matches=np.ones((1, 1)), targ_poses=[left_zone_pose], replace=False,
                          rotations=False, metric='zone', params=[(left_zone_pose, left_zone_size)], step_max_reward=goal_reward,
                          language_goal=self.lang_template.format(item="food item", side="left"))

        # Goal: Place each non-food item on the right side of the table
        for i, obj_id in enumerate(non_food_items):
            self.add_goal(objs=[obj_id], matches=np.ones((1, 1)), targ_poses=[right_zone_pose], replace=False,
                          rotations=False, metric='zone', params=[(right_zone_pose, right_zone_size)], step_max_reward=goal_reward,
                          language_goal=self.lang_template.format(item="non-food item", side="right"))

    def additional_reset(self):
        self.goals = []
        self.lang_goals = []
        self.progress = 0
        self._rewards = 0

# Ensure proper reset by overriding the reset method and fixing the issue
def reset(self, env):
    super().reset(env)
    self.additional_reset()