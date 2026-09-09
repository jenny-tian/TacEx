# RaspberryPick

Force-limited red-raspberry harvesting in a layered raspberry grove. The task
uses physical fingertip contact, a damage threshold for pad-normal grip force,
and a separate pedicle-tension threshold for detachment.

Collect one synchronized dual-camera trajectory and an annotated MP4:

```bash
/home/limx/anaconda3/envs/env_isaaclab/bin/python \
  scripts/demos/raspberry_pick/collect_dataset_video.py \
  --seed 930000 --max-episode-steps 900 --record-stride 4 \
  --output-root exp_report/raspberry_pick/data_collection \
  --video-path exp_report/raspberry_pick/raspberry_data_collection.mp4 \
  --headless
```

