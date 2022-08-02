#!/bin/bash

STRING="[*] Data Preprocessing"
echo $STRING

python preprocessing.py

STRING="[*] Start Training"
echo $STRING

cd ../
# For Single GPU (p5 model)
python train.py --workers 1 --device 0 --batch-size 16 --data sar/sar.yaml --img 256 256 --cfg cfg/training/yolov7.yaml --weights '' --name yolov7 --hyp data/hyp.scratch.p5.yaml

# For Multi-GPU (p5 model)
# python -m torch.distributed.launch --nproc_per_node 4 --master_port 9527 train.py --workers 8 --device 0,1,2,3 --sync-bn --batch-size 128 --data sar/sar.yaml --img 640 640 --cfg cfg/training/yolov7.yaml --weights '' --name yolov7 --hyp data/hyp.scratch.p5.yaml

# Inference
#python detect.py --weights ./runs/train/yolov7/weights/best.pt --conf 0.25 --img-size 256 --source inference/images/ships.jpg