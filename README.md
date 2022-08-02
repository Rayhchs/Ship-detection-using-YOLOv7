# Ship Detection From SAR Images Using YOLOv7
Thanks to incredible work of [YOLOv7](https://github.com/WongKinYiu/yolov7) ([YOLOv7 paper](https://arxiv.org/abs/2207.02696)) for object detection, this repo applies YOLOv7 model to implement ship detection from SAR images. The SAR images is derived from [SAR-Ship-Dataset](https://github.com/CAESAR-Radi/SAR-Ship-Dataset). This integral dataset is composed of 39,729 ship chips cropped from 102 Chinese Gaofen-3 images and 108 Sentinel-1 with 256 by 256 pixels. This repo seperate dataset into 90% of training, 5% of validation and 5% of testing. In the training process, we modify some configuration but mainly follow the process provided by [YOLOv7](https://github.com/WongKinYiu/yolov7). This repo provides modified config and process from data preprocessing to evaluation.

## Quick Start
* Clone [YOLOv7 model](https://github.com/WongKinYiu/yolov7)
* Download SAR dataset from [SAR-Ship-Dataset](https://github.com/CAESAR-Radi/SAR-Ship-Dataset)
* Clone this repository and move to yolov7

      git clone https://github.com/Rayhchs/Ship-detection-using-YOLOv7.git
      mv Ship-detection-using-YOLOv7 yolov7/sar
      
* Move SAR dataset to sar/

      mv SAR-Ship-Dataset/ship_dataset_v0.zip yolov7/sar/ship_dataset_v0.zip
      unzip ship_dataset_v0.zip
      
* Build environment

      cd yolov7/sar
      pip install --no-cache-dir -r requirements.txt
      
* Train YOLOv7
      
      sh train.sh
      
* Evaluation

      sh. eval.sh
      
* Additional images
Downloald cropped images from  and put it into appendix

      python inference.py --source ./appendix --weights ../runs/train/yolov7/weights/best.pt --img-size 256 --conf-thres 0.5

      
      
      
      
