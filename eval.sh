#!/bin/bash

STRING="[*] Inferencing"
echo $STRING

python inference.py --source ./images/test --weights ../runs/train/yolov7/weights/best.pt --img-size 256

STRING="[*] Evaluating"
echo $STRING

# For Single GPU (p5 model)
python evaluation.py --pred_txt ./results/ --label_txt ./labels/test/ --img_size 256

# For Multi-GPU (p5 model)
# python -m torch.distributed.launch --nproc_per_node 4 --master_port 9527 train.py --workers 8 --device 0,1,2,3 --sync-bn --batch-size 128 --data sar/sar.yaml --img 640 640 --cfg cfg/training/yolov7.yaml --weights '' --name yolov7 --hyp data/hyp.scratch.p5.yaml