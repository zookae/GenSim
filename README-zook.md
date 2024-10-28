prerequisites

- download HOPE assets
```bash
# from local
rsync -avz /mnt/c/Users/azook/projects/lc-agent/cliport/environments/assets/HOPE horde@$GENSIM:/home/horde/projects/GenSim/cliport/environments/assets/

# between devices
cp -rf ../lc-agent/cli
port/environments/assets/HOPE ./cliport/environments/assets/
```


to record videos with blender output:

```bash
export GENSIM_ROOT=$(pwd)

python cliport/demos.py n=2 task=grouping-boxes-by-category mode=test disp=False +record.blender_render=True record.save_video=True
```