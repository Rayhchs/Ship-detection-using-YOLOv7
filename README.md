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
      
Pretrain model can be download from [Here](https://drive.google.com/file/d/1-h2OShlVZhr5iXTQ8DApD4M7eLBhYJ5D/view?usp=sharing). Put the pretrain weight under yolov7/runs/train/yolov7/weights/
      
* Evaluation

      sh. eval.sh
      
* Additional images
This repository also test the pretrain model on real SAR image cropped from Sentinel 1 image. You can downloald cropped images from [Here](https://drive.google.com/drive/folders/11hxJRMBh_dd4ro-Gda9_-rIX0ImGc_g8?usp=sharing) and put it into sar\. 

      python inference.py --source ./appendix --weights ../runs/train/yolov7/weights/best.pt --img-size 256 --conf-thres 0.5

## Results
Here shows some detection results of testing set. And its' evaluation results.

<img src="https://github.com/Rayhchs/Ship-detection-using-YOLOv7/blob/main/imgs/results.jpg" width="720">
<img src="https://github.com/Rayhchs/Ship-detection-using-YOLOv7/blob/main/imgs/eval.jpg" width="720">

## Real Sentinel-1 image
We crop the SAR image from Sentinel-1 manually. However, since the training dataset doesn't include the land image, we exclude the land part of SAR image. The detected result is shown as below.

<img src="https://github.com/Rayhchs/Ship-detection-using-YOLOv7/blob/main/imgs/appendix.jpg" width="720">

## Acknowledgements
Thanks to incredible objection detection model [YOLOv7](https://github.com/WongKinYiu/yolov7) and [SAR dataset](https://github.com/CAESAR-Radi/SAR-Ship-Dataset)

If you use this code for your research, please cite their original papers:

            @article{wang2022yolov7,
              title={{YOLOv7}: Trainable bag-of-freebies sets new state-of-the-art for real-time object detectors},
              author={Wang, Chien-Yao and Bochkovskiy, Alexey and Liao, Hong-Yuan Mark},
              journal={arXiv preprint arXiv:2207.02696},
              year={2022}
            }

		@Article{rs11070765, AUTHOR = {Wang, Yuanyuan and Wang, Chao and Zhang, Hong and Dong, Yingbo and Wei, Sisi},
            TITLE = {A SAR Dataset of Ship Detection for Deep Learning under Complex Backgrounds},
            JOURNAL = {Remote Sensing},
            VOLUME = {11},
            YEAR = {2019},
            NUMBER = {7},
            ARTICLE-NUMBER = {765},
            URL = {http://www.mdpi.com/2072-4292/11/7/765},
            ISSN = {2072-4292},
            DOI = {10.3390/rs11070765}
            }
            
