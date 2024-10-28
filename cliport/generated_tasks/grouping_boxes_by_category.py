import numpy as np
from cliport.tasks.task import Task
from cliport.utils import utils
import pybullet as p

class GroupingBoxesByCategory(Task):
    """Group the boxes of macaroni and cheese, popcorn, granola bars, and raisins into separate clusters on the table."""

    def __init__(self):
        super().__init__()
        self.max_steps = 20
        self.lang_template = "group the {item} boxes together"
        self.task_completed_desc = "done grouping boxes by category."
        self.additional_reset()

    def reset(self, env):
        super().reset(env)

        # Define object categories and their URDF paths
        categories = {
            'macaroni and cheese': 'HOPE/MacaroniAndCheese.urdf',
            'popcorn': 'HOPE/Popcorn.urdf',
            'granola bars': 'HOPE/GranolaBars.urdf',
            'raisins': 'HOPE/Raisins.urdf'
        }

        # Define the bounding boxes for each category
        bounding_boxes = {
            'raisins': [
                [np.array([0.64181176, 0.09783743, 0.01725039]), np.array([0.56594454, 0.21008304, 0.10618795])],
                [np.array([0.52334808, 0.25449768, 0.00622667]), np.array([0.4588142 , 0.38106795, 0.09930298])]
            ],
            'macaroni and cheese': [
                [np.array([0.46462215, 0.03829987, 0.01004018]), np.array([0.13537785, 0.17639925, 0.16149769])],
                [np.array([ 0.71833591, -0.30722991,  0.04664086]), np.array([ 0.59652155, -0.16074623,  0.17377439])]
            ],
            'popcorn': [
                [np.array([0.4672179 , 0.0724802 , 0.00643336]), np.array([0.3733687 , 0.19056419, 0.09874854])],
                [np.array([ 0.59575359, -0.12626017,  0.01511059]), np.array([ 0.4851995 , -0.0269933 ,  0.10398766])]
            ],
            'granola bars': [
                [np.array([ 0.57950273, -0.44073914,  0.04259629]), np.array([ 0.39418153, -0.35926086,  0.1678189 ])]
            ]
        }

        objects = []
        target_poses = []

        # Add objects and define their target poses
        for category, urdf in categories.items():
            for bbox in bounding_boxes[category]:
                # Calculate the pose from the bounding box
                pos = (bbox[0] + bbox[1]) / 2
                size = bbox[1] - bbox[0]
                rot = utils.eulerXYZ_to_quatXYZW((0, 0, 0))

                # Add object
                box_pose = (pos, rot)
                box_id = env.add_object(urdf, box_pose)
                objects.append(box_id)

                # Define target pose within bounds
                target_pose = self.get_random_pose(env, size)
                while not self.is_pose_valid(target_pose, objects, size):
                    target_pose = self.get_random_pose(env, size)
                target_poses.append(target_pose)

                # Add goal for each object
                language_goal = self.lang_template.format(item=category)
                self.add_goal(objs=[box_id], matches=np.int32([[1]]), targ_poses=[target_pose], replace=False,
                              rotations=True, metric='pose', params=None, step_max_reward=1 / len(categories),
                              symmetries=[np.pi/2], language_goal=language_goal)

    def is_pose_valid(self, pose, existing_objs, obj_size):
        """Check if the pose is valid and collision-free."""
        for obj_id in existing_objs:
            obj_pose = p.getBasePositionAndOrientation(obj_id)
            if self.is_collision(pose, obj_pose, obj_size):
                return False
        return True

    def is_collision(self, pose1, pose2, obj_size):
        """Check if two poses are in collision."""
        dist = np.linalg.norm(np.array(pose1[0]) - np.array(pose2[0]))
        return dist < np.linalg.norm(obj_size)