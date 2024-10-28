import numpy as np
from cliport.tasks.task import Task
from cliport.utils import utils
import pybullet as p

class GroupItemsByCategory(Task):
    """Pick up the boxes and group them by category (snacks, beverages, pasta) on the tabletop."""

    def __init__(self):
        super().__init__()
        self.max_steps = 9
        self.lang_template = "group the {category} items together"
        self.task_completed_desc = "done grouping items by category."
        self.additional_reset()

    def reset(self, env):
        super().reset(env)
        
        # Define the items and their categories with bounding boxes.
        items = {
            'snacks': [
                ('HOPE/Raisins.urdf', [np.array([0.68920594, -0.01223175, -0.01545314]), np.array([0.60273017, 0.11508146, 0.01710281])]),
                ('HOPE/GranolaBars.urdf', [np.array([0.75999709, -0.14352535, -0.00297771]), np.array([0.64000291, 0.02287287, 0.02234672])]),
                ('HOPE/Cookies.urdf', [np.array([0.52343836, 0.07625133, 0.01987186]), np.array([0.24349915, 0.32384748, 0.20559445])]),
                ('HOPE/Popcorn.urdf', [np.array([0.56025228, 0.09958157, -0.00858464]), np.array([0.4164944, 0.19137291, 0.08246814])])
            ],
            'beverages': [
                ('HOPE/OrangeJuice.urdf', [np.array([0.68052253, 0.13157798, 0.0513954]), np.array([0.57575903, 0.2344425, 0.2422972])])
            ],
            'pasta': [
                ('HOPE/MacaroniAndCheese.urdf', [np.array([0.48742566, -0.22607871, 0.04633672]), np.array([0.41558414, -0.10495971, 0.2163609])]),
                ('HOPE/Spaghetti.urdf', [np.array([0.65748574, -0.52529668, 0.00845122]), np.array([0.61535348, -0.27470332, 0.06104281])])
            ]
        }
        
        # Define target zones for each category.
        zone_size = (0.1, 0.1, 0)
        snack_zone_pose = self.get_random_pose(env, zone_size)
        beverage_zone_pose = self.get_random_pose(env, zone_size)
        pasta_zone_pose = self.get_random_pose(env, zone_size)

        # Add target zones to the environment.
        env.add_object('zone/zone.urdf', snack_zone_pose, 'fixed')
        env.add_object('zone/zone.urdf', beverage_zone_pose, 'fixed')
        env.add_object('zone/zone.urdf', pasta_zone_pose, 'fixed')

        # Add objects to the environment and define their goals.
        objs, goals = [], []
        for category, urdf_boxes in items.items():
            for urdf, bbox in urdf_boxes:
                size = (bbox[1] - bbox[0])  # Calculate size from bounding box
                pose = ((bbox[0] + bbox[1]) / 2, (0, 0, 0, 1))  # Center of bounding box
                obj_id = env.add_object(urdf, pose)
                objs.append(obj_id)
                
                # Define the goal for each item based on its category.
                if category == 'snacks':
                    target_pose = snack_zone_pose
                elif category == 'beverages':
                    target_pose = beverage_zone_pose
                else:
                    target_pose = pasta_zone_pose
                
                self.add_goal(objs=[obj_id], matches=np.int32([[1]]), targ_poses=[target_pose], replace=False,
                              rotations=True, metric='zone', params=[(target_pose, zone_size)], step_max_reward=1/len(urdf_boxes),
                              language_goal=self.lang_template.format(category=category))