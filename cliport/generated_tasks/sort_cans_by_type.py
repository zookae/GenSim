import numpy as np
from cliport.tasks.task import Task
from cliport.utils import utils
import pybullet as p

class SortCansByType(Task):
    """Pick up the cans and sort them into groups based on their type."""

    def __init__(self):
        super().__init__()
        self.max_steps = 15
        self.lang_template = "sort the {type} cans into the {color} zone"
        self.task_completed_desc = "done sorting cans."
        self.additional_reset()

    def reset(self, env):
        super().reset(env)

        # Define can types and their corresponding URDF files
        can_types = {
            'soup': 'HOPE/AlphabetSoup.urdf',
            'vegetables': 'HOPE/PeasAndCarrots.urdf',
            'tomato sauce': 'HOPE/TomatoSauce.urdf',
            'parmesan': 'HOPE/Parmesan.urdf',
            'yogurt': 'HOPE/Yogurt.urdf',
            'corn': 'HOPE/Corn.urdf',
            'granola bars': 'HOPE/GranolaBars.urdf'
        }

        # Predefined bounding boxes
        bounding_boxes = {
            'HOPE/AlphabetSoup.urdf': [np.array([0.56462766, -0.13425138, 0.00282482]), np.array([0.48782656, -0.07136663, 0.08508737])],
            'HOPE/GranolaBars.urdf': [np.array([0.63596572, 0.26615694, 0.00760562]), np.array([0.46387101, 0.33705992, 0.13629785])],
            'HOPE/TomatoSauce.urdf': [np.array([0.41437349, -0.20947873, 0.00281065]), np.array([0.30770377, -0.1406497, 0.09852502])],
            'HOPE/Parmesan.urdf': [np.array([0.50209746, -0.04205837, 0.00437512]), np.array([0.32939386, 0.04772601, 0.11185423])],
            'HOPE/Yogurt.urdf': [np.array([0.73733665, -0.13262174, -0.00331223]), np.array([0.66266335, -0.06884041, 0.04882852])],
            'HOPE/PeasAndCarrots.urdf': [np.array([0.70739611, -0.22923575, 0.00099086]), np.array([0.63098337, -0.16512773, 0.0558798])],
            'HOPE/Corn.urdf': [np.array([0.51681334, -0.29184396, -0.00544687]), np.array([0.43276236, -0.21779869, 0.06054364])]
        }

        # Colors for zones
        zone_colors = {
            'soup': [1, 0, 0, 1],  # Red
            'vegetables': [0, 1, 0, 1],  # Green
            'tomato sauce': [0, 0, 1, 1],  # Blue
            'parmesan': [1, 1, 0, 1],  # Yellow
            'yogurt': [1, 0, 1, 1],  # Magenta
            'corn': [0, 1, 1, 1],  # Cyan
            'granola bars': [1, 0.5, 0, 1]  # Orange
        }

        # Randomly position the cans in the environment
        cans = []
        for can_type, urdf in can_types.items():
            if urdf in bounding_boxes:
                # Use predefined bounding boxes
                can_pose = (
                    (bounding_boxes[urdf][0] + bounding_boxes[urdf][1]) / 2,
                    utils.eulerXYZ_to_quatXYZW([0, 0, np.random.rand() * 2 * np.pi])
                )
                can_id = env.add_object(urdf, can_pose)
                cans.append((can_type, can_id))

        # Create sorting zones for each type of can
        zones = {}
        for can_type in can_types.keys():
            zone_size = (0.12, 0.12, 0.001)
            zone_pose = self.get_random_pose(env, zone_size)
            zone_urdf = 'zone/zone.urdf'
            zone_id = env.add_object(zone_urdf, zone_pose, 'fixed')
            p.changeVisualShape(zone_id, -1, rgbaColor=zone_colors[can_type])
            zones[can_type] = zone_pose

        # Add goals for sorting the cans into the correct zones
        for can_type, can_id in cans:
            zone_pose = zones[can_type]
            language_goal = self.lang_template.format(type=can_type, color=can_type)
            self.add_goal(
                objs=[can_id],
                matches=np.int32([[1]]),
                targ_poses=[zone_pose],
                replace=False,
                rotations=True,
                metric='zone',
                params=[(zone_pose, (0.12, 0.12, 0.001))],
                step_max_reward=1 / len(cans),
                language_goal=language_goal
            )