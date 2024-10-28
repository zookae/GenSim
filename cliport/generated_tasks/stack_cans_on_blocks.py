import numpy as np
import pybullet as p
from cliport.tasks import primitives
from cliport.tasks.grippers import Spatula
from cliport.tasks.task import Task
from cliport.utils import utils
import pybullet_utils

class StackCansOnBlocks(Task):
    """Pick up the scattered cans and stack each one on top of a block of matching height."""

    def __init__(self):
        super().__init__()
        self.max_steps = 10
        self.lang_template = "stack the {can} on top of the {block}"
        self.task_completed_desc = "done stacking cans on blocks."
        self.additional_reset()

    def reset(self, env):
        super().reset(env)

        # Define the blocks and their fixed poses based on provided bounding boxes
        block_urdf = 'block/block.urdf'
        block_colors = ['red', 'green', 'blue', 'yellow', 'purple', 'orange']
        blocks_poses = [
            ((0.6899593 + 0.63709411) / 2, (0.04932231 + -0.00730875) / 2, 0.04696857 / 2),
            ((0.41495827 + 0.34571097) / 2, (0.26286784 + 0.34013691) / 2, 0.03010734 / 2),
            ((0.52125684 + 0.43918107) / 2, (0.30908895 + 0.38713542) / 2, 0.03025877 / 2),
            ((0.70430812 + 0.65980438) / 2, (-0.23231308 + -0.17999047) / 2, 0.05874907 / 2),
            ((0.60029041 + 0.53093714) / 2, (0.267829 + 0.34830929) / 2, 0.03652433 / 2),
            ((0.34947348 + 0.2523094) / 2, (0.05951131 + 0.13256332) / 2, 0.02898239 / 2)
        ]
        blocks = []

        # Add blocks to the environment
        for i, color in enumerate(block_colors):
            block_pose = ((blocks_poses[i][0], blocks_poses[i][1], blocks_poses[i][2]), (0, 0, 0, 1))
            block_id = env.add_object(block_urdf, block_pose, category='fixed')
            p.changeVisualShape(block_id, -1, rgbaColor=utils.COLORS[color])
            blocks.append(block_id)

        # Define the cans and their fixed poses based on provided bounding boxes
        cans_urdfs = [
            'HOPE/AlphabetSoup.urdf',
            'HOPE/Tuna.urdf',
            'HOPE/Yogurt.urdf',
            'HOPE/Peaches.urdf',
            'HOPE/PeasAndCarrots.urdf'
        ]
        cans_poses = [
            ((0.42359892 + 0.35004971) / 2, (0.12288328 + 0.22160096) / 2, 0.05028934 / 2),
            ((0.52154596 + 0.45653369) / 2, (0.11828353 + 0.19151019) / 2, 0.05628396 / 2),
            ((0.56525667 + 0.48928761) / 2, (-0.34408256 + -0.27805748) / 2, 0.05465149 / 2),
            ((0.48019342 + 0.39529506) / 2, (-0.18347392 + -0.11947069) / 2, 0.05885765 / 2),
            ((0.48735866 + 0.39408506) / 2, (-0.31742158 + -0.23030524) / 2, 0.06749361 / 2)
        ]
        cans = []

        # Add cans to the environment
        for i, can_urdf in enumerate(cans_urdfs):
            can_pose = ((cans_poses[i][0], cans_poses[i][1], cans_poses[i][2]), (0, 0, 0, 1))
            can_id = env.add_object(can_urdf, can_pose)
            cans.append(can_id)

        # Create goals
        for i in range(len(cans)):
            block_pose = p.getBasePositionAndOrientation(blocks[i])
            can_size = (0.05, 0.05, 0.1)
            can_target_pose = ((block_pose[0][0], block_pose[0][1], block_pose[0][2] + 0.04 / 2 + can_size[2] / 2), block_pose[1])
            
            language_goal = self.lang_template.format(can=cans_urdfs[i].split('/')[1].split('.')[0], block=block_colors[i])
            self.add_goal(
                objs=[cans[i]],
                matches=np.int32([[1]]),
                targ_poses=[can_target_pose],
                replace=False,
                rotations=True,
                metric='pose',
                params=None,
                step_max_reward=1 / len(cans),
                language_goal=language_goal
            )