import os
import numpy as np
from cliport.tasks.task import Task
from cliport.utils import utils
import pybullet as p

class CategorizeFoodItems(Task):
    """Pick up each food item and place it into one of two specific zones on the table: one for boxed items and one for canned items."""

    def __init__(self):
        super().__init__()
        self.max_steps = 16
        self.lang_template = "place the {item} in the {zone} zone"
        self.task_completed_desc = "done categorizing food items."
        self.additional_reset()

    def reset(self, env):
        super().reset(env)

        # Define assets
        boxed_items = ['HOPE/Parmesan.urdf', 'HOPE/MacaroniAndCheese.urdf', 'HOPE/OrangeJuice.urdf', 
                       'HOPE/Popcorn.urdf', 'HOPE/GranolaBars.urdf']
        canned_items = ['HOPE/Corn.urdf', 'HOPE/AlphabetSoup.urdf', 'HOPE/TomatoSauce.urdf']
        all_items = boxed_items + canned_items

        # Define bounding boxes
        bounding_boxes = {
            'HOPE/Parmesan.urdf': [np.array([0.67525025, 0.21736709, 0.01479784]), np.array([0.60144924, 0.28641626, 0.11919077])],
            'HOPE/MacaroniAndCheese.urdf': [np.array([0.59935164, 0.31777381, 0.04268818]), np.array([0.50789378, 0.47693904, 0.20365712])],
            'HOPE/OrangeJuice.urdf': [np.array([0.4492064 , 0.15927754, 0.03436354]), np.array([0.1507936 , 0.2961691 , 0.23876241])],
            'HOPE/Popcorn.urdf': [np.array([ 0.47812411, -0.26758373,  0.00665536]), np.array([ 0.41712827, -0.14386012,  0.09590175])],
            'HOPE/Corn.urdf': [np.array([ 0.67271801,  0.09835504, -0.0069891 ]), np.array([0.59845303, 0.17080098, 0.05410966])],
            'HOPE/AlphabetSoup.urdf': [np.array([ 0.67044673, -0.07752964,  0.00729417]), np.array([ 0.58047916, -0.01311518,  0.08906239])],
            'HOPE/TomatoSauce.urdf': [np.array([ 0.46984756, -0.04067392, -0.00194045]), np.array([0.38206558, 0.02383856, 0.07872978])],
            'HOPE/GranolaBars.urdf': [np.array([ 0.56771373, -0.43796624,  0.03478694]), np.array([ 0.49924767, -0.2662086 ,  0.16232731])]
        }

        # Define zones
        zone_size = (0.2, 0.2, 0.02)
        box_zone_pose = self.get_random_pose(env, zone_size)
        can_zone_pose = self.get_random_pose(env, zone_size)

        # Add zones to the environment
        box_zone_urdf = 'zone/zone.urdf'
        can_zone_urdf = 'zone/zone.urdf'

        env.add_object(box_zone_urdf, box_zone_pose, 'fixed')
        env.add_object(can_zone_urdf, can_zone_pose, 'fixed')

        # Add objects to the environment
        objects = []
        for urdf in all_items:
            bbox = bounding_boxes[urdf]
            obj_pose = (bbox[0] + bbox[1]) / 2, utils.eulerXYZ_to_quatXYZW((0, 0, 0))
            obj_id = env.add_object(urdf, obj_pose, 'rigid')
            objects.append((obj_id, urdf))

        # Define goals
        for obj_id, urdf in objects:
            if urdf in boxed_items:
                zone_pose = box_zone_pose
                zone_name = "boxed items"
            else:
                zone_pose = can_zone_pose
                zone_name = "canned items"
            
            language_goal = self.lang_template.format(item=urdf.split('/')[1].split('.')[0], zone=zone_name)
            self.add_goal(objs=[obj_id], matches=np.int32([[1]]), targ_poses=[zone_pose], replace=False,
                          rotations=True, metric='zone', params=[(zone_pose, zone_size)], step_max_reward=1/len(objects),
                          language_goal=language_goal)