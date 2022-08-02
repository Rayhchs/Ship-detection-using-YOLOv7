import argparse
from glob2 import glob
import os
from tqdm import tqdm
import numpy as np

def evaluation():
    pred_txts, label_txts, sz = opt.pred_txt, opt.label_txt, opt.img_size
    pred_txts = glob(pred_txts + '**.txt')
    label_tmp = glob(label_txts + '**.txt')
    
    print('[*] check txt file')
    label_txts = []
    for i in tqdm(label_tmp):
        basename_label = os.path.basename(i)
        for j in pred_txts:
            basename = os.path.basename(j)[:-10]
            if basename+'.txt' in basename_label:
                label_txts.append(i)
    
    assert len(pred_txts) == len(label_txts)

    
    mses = []
    tp, fp, fn, tn = 0, 0, 0, 0
    for i in range(len(pred_txts)):
        with open(pred_txts[i], 'r') as f:
            preds = f.readlines()

        with open(label_txts[i], 'r') as f:
            trues = f.readlines()
            
        if len(preds) > len(trues):
            fp += (len(preds) - len(trues))
        elif len(preds) < len(trues):
            fn += (len(trues) - len(preds))
        elif len(preds) == 0 and len(trues) == 0:
            tn += 1            
        elif len(preds) == len(trues):
            tp += len(preds)
            
            for j in range(len(preds)):
                pred = np.array(preds[j].strip().split(' ')[1:5]).astype(np.float32)
                true = np.array(trues[j].strip().split(' ')[1:5]).astype(np.float32)
                mses.append(np.square(pred-true))
    
    x0, y0, x1, y1 = 0, 0, 0, 0
    for i in range(len(mses)):
        x0 += mses[i][0]
        y0 += mses[i][1]
        x1 += mses[i][2]
        y1 += mses[i][3]
    x0, y0, x1, y1 = x0/len(mses)*sz, y0/len(mses)*sz, x1/len(mses)*sz, y1/len(mses)*sz
    
    print(f'TP: {tp}; FP: {fp}; FN: {fn}; Distant-x0: {x0}; y0: {y0}; x1: {x1}; y1: {y1} (pixels)')
            

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred_txt', type=str, default='./results/', help='folder contains predicted txt file')
    parser.add_argument('--label_txt', type=str, default='./labels/test/', help='folder contains label txt file')
    parser.add_argument('--img_size', type=int, default=256, help='origin image size')
    opt = parser.parse_args()
    evaluation()