import argparse
import time
from pathlib import Path
import tifffile as tiff
from glob2 import glob
import cv2, sys, os
import torch
import torch.backends.cudnn as cudnn
from numpy import random
import numpy as np
from tqdm import tqdm

prepath, _ = os.path.split(os.getcwd())
sys.path.append(prepath)

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel


def crop(source):
    path, imgsz = source, opt.img_size
    if 'tif' in path or 'tiff' in path:
        img = tiff.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    else:
        img = cv2.imread(path)
    img = np.float32(img)

    w_num = int(img.shape[0]/imgsz) if img.shape[0] % imgsz != 0 else int(img.shape[0]/imgsz)-1
    h_num = int(img.shape[1]/imgsz) if img.shape[1] % imgsz != 0 else int(img.shape[1]/imgsz)-1

    imgs = []
    for i in range(w_num+1):
        for j in range(h_num+1):
            if i == w_num and j != h_num:
                imgs.append(img[img.shape[0]-imgsz:img.shape[0], imgsz*j:imgsz*(j+1), :])
            elif j == h_num and i != w_num:
                imgs.append(img[imgsz*i:imgsz*(i+1), img.shape[1]-imgsz:img.shape[1], :])
            elif j == h_num and i == w_num:
                imgs.append(img[img.shape[0]-imgsz:img.shape[0], img.shape[1]-imgsz:img.shape[1], :])
            else:
                imgs.append(img[imgsz*i:imgsz*(i+1), imgsz*j:imgsz*(j+1), :])

    return imgs


def merge(imgs, source):
    path, imgsz = source, opt.img_size
    if 'tif' in path or 'tiff' in path:
        img = tiff.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    else:
        img = cv2.imread(path)

    w_num = int(img.shape[0]/imgsz) if img.shape[0] % imgsz != 0 else int(img.shape[0]/imgsz)-1
    h_num = int(img.shape[1]/imgsz) if img.shape[1] % imgsz != 0 else int(img.shape[1]/imgsz)-1
    
    newimg = img.copy()
    for i in range(w_num+1):
        for j in range(h_num+1):
            idx = (i*(h_num+1))+(j)
            if i == w_num and j != h_num:
                newimg[img.shape[0]-imgsz:img.shape[0], imgsz*j:imgsz*(j+1), :] = imgs[idx]
            elif j == h_num and i != w_num:
                newimg[imgsz*i:imgsz*(i+1), img.shape[1]-imgsz:img.shape[1], :] = imgs[idx]
            elif j == h_num and i == w_num:
                newimg[img.shape[0]-imgsz:img.shape[0], img.shape[1]-imgsz:img.shape[1], :] = imgs[idx]
            else:
                newimg[imgsz*i:imgsz*(i+1), imgsz*j:imgsz*(j+1), :] = imgs[idx]
                
    return newimg


def detect(save_img=True):

    os.mkdir('./results') if not os.path.exists('./results') else None
    weights, save_txt, imgsz, trace = opt.weights, opt.save_txt, opt.img_size, not opt.no_trace
    sources = glob(opt.source + '/**.jpg') + glob(opt.source + '/**.png') + glob(opt.source + '/**.tif') + glob(opt.source + '/**.tiff')
    #save_img = not opt.nosave and not source.endswith('.txt')  # save inference images

    # Initialize
    set_logging()
    device = select_device(opt.device)
    half = False#device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size

    if trace:
        model = TracedModel(model, device, opt.img_size)

    if half:
        model.half()  # to FP16

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    t0 = time.time()

    for source in tqdm(sources):

        # Crop image into piece because remote sensing image is too large
        dataset = crop(source)

        filename = os.path.basename(source)
        detected_imgs = []
        for img in tqdm(dataset):

            im0 = img.copy()

            img = np.transpose(img, [2, 0, 1])
            img = torch.from_numpy(img).to(device)
            img = img.half() if half else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)

            # Inference
            t1 = time_synchronized()
            pred = model(img, augment=opt.augment)[0]

            # Apply NMS
            pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
            t2 = time_synchronized()

            # Process detections
            for i, det in enumerate(pred):  # detections per image

                save_path = './results/'
                txtname, _ = filename.split('.')
                txt_path = './results/' + txtname + '_label.txt'
                gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh

                if len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                    # Print results
                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()  # detections per class

                    # Write results
                    for *xyxy, conf, cls in reversed(det):
                        if save_txt:  # Write to file
                            xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                            line = (cls, *xywh, conf) if opt.save_conf else (cls, *xywh)  # label format
                            with open(txt_path, 'a') as f:
                                f.write(('%g ' * len(line)).rstrip() % line + '\n')
                        if conf >= opt.conf_thres:
                            if save_img or view_img:  # Add bbox to image
                                label = f'{names[int(cls)]} {conf:.2f}'
                                plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)

            detected_imgs.append(im0)
        result = merge(detected_imgs, source)

        if save_img:
            if 'tif' in filename or 'tiff' in filename:
               cv2.imwrite(save_path + txtname + '.jpg', result) 
            cv2.imwrite(save_path + filename, result)
            print(f" The image with the result is saved in: {save_path}")


    print(f'Done. ({time.time() - t0:.3f}s)')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov7.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='inference/images', help='source folder')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--save-txt', default=True, action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', default=True, action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes',nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--no-trace', action='store_true', help='don`t trace model')
    opt = parser.parse_args()
    #check_requirements(exclude=('pycocotools', 'thop'))
    print(opt)
    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['yolov7.pt']:
                detect()
                strip_optimizer(opt.weights)
        else:
            detect()