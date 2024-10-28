import numpy as np
from cliport.tasks.task import Task
from cliport.utils import utils
import pybullet as p


class OrganizeFoodItems(Task):
    """Pick up and organize all food items into two groups: beverages and snacks, placing them in separate designated zones on the tabletop."""

    def __init__(self):
        super().__init__()
        self.max_steps = 15  # Increase max steps to give the agent more time
        self.lang_template = "organize the food items, placing the {item} in the designated zone"
        self.task_completed_desc = "done organizing food items."
        self.additional_reset()

    def reset(self, env):
        super().reset(env)

        # Food item URDFs and their bounding boxes
        food_items = [
            ('HOPE/OrangeJuice.urdf', [np.array([0.65492414, 0.13401552, 0.0513954]), np.array([0.55016065, 0.23688004, 0.2422972])]),
            ('HOPE/Raisins.urdf', [np.array([0.66399987, -0.01034437, -0.01545314]), np.array([0.5775241, 0.11696885, 0.01710281])]),
            ('HOPE/GranolaBars.urdf', [np.array([0.73598009, -0.14210519, -0.00297771]), np.array([0.6159859, 0.02429303, 0.02234672])]),
            ('HOPE/MacaroniAndCheese.urdf', [np.array([0.4579401, -0.22509836, 0.04633672]), np.array([0.38609859, -0.10397936, 0.2163609])]),
            ('HOPE/Popcorn.urdf', [np.array([0.53157808, 0.10186218, -0.00858464]), np.array([0.3878202, 0.19365352, 0.08246814])]),
            ('HOPE/Spaghetti.urdf', [np.array([0.63206954, -0.52529668, 0.00845122]), np.array([0.58993728, -0.27470332, 0.06104281])]),
            ('HOPE/Cookies.urdf', [np.array([0.49245556, 0.0787601, 0.01987186]), np.array([0.21251635, 0.32635625, 0.20559445])])
        ]

        # Define the beverage and snack categories
        beverages = ['HOPE/OrangeJuice.urdf']
        snacks = ['HOPE/Raisins.urdf', 'HOPE/GranolaBars.urdf', 'HOPE/MacaroniAndCheese.urdf', 
                  'HOPE/Popcorn.urdf', 'HOPE/Spaghetti.urdf', 'HOPE/Cookies.urdf']

        # Add zones for beverages and snacks
        beverage_zone_size = (0.2, 0.2, 0.02)
        snack_zone_size = (0.2, 0.2, 0.02)
        beverage_zone_pose = self.get_random_pose(env, beverage_zone_size)
        snack_zone_pose = self.get_random_pose(env, snack_zone_size)

        # Add fixed zones to the environment
        env.add_object('zone/zone.urdf', beverage_zone_pose, 'fixed')
        env.add_object('zone/zone.urdf', snack_zone_pose, 'fixed')

        # Add food items to the environment
        item_ids = []
        for item, bbox in food_items:
            item_size = (bbox[1][0] - bbox[0][0], bbox[1][1] - bbox[0][1], bbox[1][2] - bbox[0][2])
            item_pose = ((bbox[0] + bbox[1]) / 2, utils.eulerXYZ_to_quatXYZW((0, 0, 0)))  # Center of bounding box as position
            item_id = env.add_object(item, item_pose)
            item_ids.append(item_id)

        # Create goals for beverages and snacks
        beverage_item_ids = [item_ids[i] for i, (item, _) in enumerate(food_items) if item in beverages]
        snack_item_ids = [item_ids[i] for i, (item, _) in enumerate(food_items) if item in snacks]

        # Goal: Place beverages in the beverage zone
        self.add_goal(objs=beverage_item_ids, matches=np.ones((len(beverage_item_ids), 1)), targ_poses=[beverage_zone_pose], replace=False,
                      rotations=True, metric='zone', params=[(beverage_zone_pose, beverage_zone_size)], step_max_reward=1.0 / len(beverage_item_ids), 
                      language_goal=self.lang_template.format(item='beverages'))

        # Goal: Place snacks in the snack zone
        self.add_goal(objs=snack_item_ids, matches=np.ones((len(snack_item_ids), 1)), targ_poses=[snack_zone_pose], replace=False,
                      rotations=True, metric='zone', params=[(snack_zone_pose, snack_zone_size)], step_max_reward=1.0 / len(snack_item_ids), 
                      language_goal=self.lang_template.format(item='snacks'))