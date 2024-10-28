import numpy as np
from cliport.tasks.task import Task
from cliport.utils import utils
import pybullet as p

class PackGroceries(Task):
    """Pick up all grocery items and neatly place them into the container."""

    def __init__(self):
        super().__init__()
        self.max_steps = 10
        self.lang_template = "pack the {item} into the container"
        self.task_completed_desc = "done packing groceries."
        self.additional_reset()

    def reset(self, env):
        super().reset(env)

        # Define the grocery items and their URDF files with predefined positions
        groceries = [
            {'name': 'Granola Bars', 'urdf': 'HOPE/GranolaBars.urdf', 'pose': (np.array([0.7, 0.1, 0.02]), np.array([0, 0, 0, 1]))},
            {'name': 'Macaroni And Cheese', 'urdf': 'HOPE/MacaroniAndCheese.urdf', 'pose': (np.array([0.35, 0.4, 0.12]), np.array([0, 0, 0, 1]))},
            {'name': 'Orange Juice', 'urdf': 'HOPE/OrangeJuice.urdf', 'pose': (np.array([0.48, 0.08, 0.15]), np.array([0, 0, 0, 1]))},
            {'name': 'Raisins', 'urdf': 'HOPE/Raisins.urdf', 'pose': (np.array([0.38, -0.07, 0.08]), np.array([0, 0, 0, 1]))},
            {'name': 'Mustard', 'urdf': 'HOPE/Mustard.urdf', 'pose': (np.array([0.55, 0.26, 0.01]), np.array([0, 0, 0, 1]))},
            {'name': 'Cookies', 'urdf': 'HOPE/Cookies.urdf', 'pose': (np.array([0.55, -0.37, 0.15]), np.array([0, 0, 0, 1]))},
        ]

        # Add container box at the predefined position
        container_size = (0.2, 0.2, 0.1)
        container_pose = (np.array([0.55, -0.12, 0.07]), np.array([0, 0, 0, 1])) # This position is derived from the bounding box provided
        container_template = 'container/container-template.urdf'
        replace = {'DIM': container_size, 'HALF': (container_size[0] / 2, container_size[1] / 2, container_size[2] / 2)}
        container_urdf = self.fill_template(container_template, replace)
        env.add_object(container_urdf, container_pose, 'fixed')

        # Add groceries to the environment
        objects = []
        for grocery in groceries:
            grocery_pose = grocery['pose']
            grocery_id = env.add_object(grocery['urdf'], grocery_pose)
            objects.append({'id': grocery_id, 'name': grocery['name']})

        # Define goals for each grocery item
        for obj in objects:
            language_goal = self.lang_template.format(item=obj['name'])
            self.add_goal(objs=[obj['id']], matches=np.int32([[1]]), targ_poses=[container_pose], replace=False, 
                          rotations=True, metric='zone', params=[(container_pose, container_size)], step_max_reward=1/len(objects), 
                          language_goal=language_goal)