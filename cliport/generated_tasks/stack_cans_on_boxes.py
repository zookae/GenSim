import numpy as np
from cliport.tasks.task import Task
from cliport.utils import utils
import pybullet as p


class StackCansOnBoxes(Task):
    """Pick up the cans and stack them on top of the boxes, ensuring each can is stably placed and balanced on a box."""

    def __init__(self):
        super().__init__()
        self.max_steps = 10
        self.lang_template = "stack the can on the box"
        self.task_completed_desc = "done stacking cans on boxes."
        self.additional_reset()

    def reset(self, env):
        super().reset(env)

        # Define box and can assets with their bounding boxes
        boxes = [
            ('HOPE/Parmesan.urdf', [np.array([0.69099942, 0.17097755, 0.01479784]), np.array([0.6171984 , 0.24002672, 0.11919077])]),
            ('HOPE/MacaroniAndCheese.urdf', [np.array([0.61115702, 0.26103279, 0.04268818]), np.array([0.51969916, 0.42019802, 0.20365712])]),
            ('HOPE/OrangeJuice.urdf', [np.array([0.4492064 , 0.11460785, 0.03436354]), np.array([0.1507936 , 0.25149942, 0.23876241])]),
            ('HOPE/Popcorn.urdf', [np.array([0.48499567, -0.28140883, 0.00665536]), np.array([0.42399983, -0.15768522, 0.09590175])]),
            ('HOPE/Cookies.urdf', [np.array([0.4789468 , 0.16703579, 0.03808662]), np.array([0.35348398, 0.34532648, 0.21379816])]),
            ('HOPE/GranolaBars.urdf', [np.array([0.57858155, -0.44137577, 0.03478694]), np.array([0.5101155 , -0.26961813, 0.16232731])])
        ]
        cans = [
            ('HOPE/Corn.urdf', [np.array([0.68833851, 0.06031371, -0.0069891]), np.array([0.61407353, 0.13275965, 0.05410966])]),
            ('HOPE/AlphabetSoup.urdf', [np.array([0.68559606, -0.10276901, 0.00729417]), np.array([0.59562848, -0.03835454, 0.08906239])]),
            ('HOPE/TomatoSauce.urdf', [np.array([0.47571047, -0.06853948, -0.00194045]), np.array([0.38792848, -0.00402699, 0.07872978])])
        ]

        # Randomly shuffle boxes and cans
        np.random.shuffle(boxes)
        np.random.shuffle(cans)

        # Add boxes to the environment
        box_poses = []
        for box, bounding_box in boxes:
            box_pose = ((bounding_box[0] + bounding_box[1]) / 2).tolist(), [0, 0, 0, 1]
            env.add_object(box, box_pose, category='rigid')
            box_poses.append(box_pose)

        # Add cans to the environment
        cans_ids = []
        for can, bounding_box in cans:
            can_pose = ((bounding_box[0] + bounding_box[1]) / 2).tolist(), [0, 0, 0, 1]
            can_id = env.add_object(can, can_pose, category='rigid')
            cans_ids.append(can_id)

        # Create goals for stacking cans on boxes
        for i in range(len(cans)):
            self.add_goal(
                objs=[cans_ids[i]], 
                matches=np.int32([[1]]), 
                targ_poses=[box_poses[i]], 
                replace=False, 
                rotations=True, 
                metric='pose', 
                params=None, 
                step_max_reward=1/len(cans), 
                symmetries=[np.pi/2], 
                language_goal=self.lang_template
            )