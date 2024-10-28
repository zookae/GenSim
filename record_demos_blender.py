import os
import subprocess

# List of task values
task_values = [
   'stack-cans-on-boxes',
   'pack-groceries',
   'sort-items-by-category',
   'organize-food-items',
   'group-items-by-category',
   'sort-items-by-category',
   'categorize-and-stack',
   'stack-cans-on-blocks',
   'sort-food-items',
   'organize-food-boxes-and-blocks',
   'sort-cans-by-type',
   'sort-and-stack-blocks',
   'sort-cans-by-color',
   'sort-food-packages',
   'stack-colorful-blocks',
   'sort-cans-by-type',
   'group-and-stack-blocks',
   'group-similar-items',
   'organize-groceries',
   'grouping-boxes-by-category',
   'sort-boxes-by-height',
   'stacking-blocks-on-packages',
   'packing-grocery-items',
   'stack-cans-on-blocks',
   'organize-kitchen-items',
   'organize-cans-by-type',
   'categorize-food-items',
   'arrange-items-by-category'
]


# Check which tasks are done
not_done_tasks = []

for task in task_values:
    path = f'./data/{task}-test'
    if not os.path.exists(path):
        not_done_tasks.append(task)

# Run the command for each task that is not done
for task in not_done_tasks:
    command = [
        'python', 'cliport/demos.py',
        f'n=3', f'task={task}', 'mode=test', 'disp=False',
        '+record.blender_render=True', 'record.save_video=True'
    ]
    
    # Execute the command
    subprocess.run(command)