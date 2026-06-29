import numpy as np
import os
import cv2
from PIL import Image
import pandas as pd
from astropy.io import fits
from scipy.spatial import KDTree
from model.utils import *
from model.metric import *
from model.loss import *
from tqdm import tqdm
import time

if __name__ == '__main__':
    count = 0
    pre_sum = 0
    rec_sum = 0
    f1_sum = 0
    fa_sum = 0
    # static snr
    bins = 15
    tp_count_array = np.zeros(bins + 1)
    fn_count_array = np.zeros(bins + 1)
    dataset='new_data_ast'
    # pred output ipd dir
    ipd_out_dir = os.path.join('/home/lixy/workspace/cdnnet/cdnnet-231010/cdnnet/dataset', dataset, 'ipd_gauss_label_test')
    label_ipd_path='/home/lixy/workspace/cdnnet/cdnnet-231010/cdnnet/dataset/new_data_ast/masks'
    input_label_ipd_dir=os.listdir(label_ipd_path)
    a=time.time()
    test_txt_path='/home/lixy/workspace/cdnnet/cdnnet-231010/cdnnet/dataset/new_data_ast/train_test/test.txt'
    filenames = []

    # 打开并读取txt文件
    with open(test_txt_path, 'r') as file:
        for line in file:
            # 去掉每行的换行符和可能存在的空格，并添加到列表中
            filenames.append(line.strip())


    for file in tqdm(filenames):
        # print(time.time()-a)
        filename=file+'.png'
        input_path=os.path.join(label_ipd_path,filename)
        image = Image.open(input_path).convert("L")
        result_array = np.array(image)
        image_array = np.zeros_like(result_array)
        image_array[result_array>=0.1]=255
        image_array[result_array<0.1]=0
        ipd_fname = os.path.join(ipd_out_dir, filename.replace('.png','.IPD'))
        ipd_file = open(ipd_fname, 'w')
        # print(time.time()-a)
        test_fitsimg = fits.getdata(os.path.join('dataset', dataset, 'fits', filename.replace('.png','.fits')))
        pred_region = measure.regionprops(measure.label(image_array, connectivity=2), intensity_image=test_fitsimg)
        for pred_i in pred_region:
            if not np.isnan(pred_i.centroid_weighted[1]) and not np.isnan(pred_i.centroid_weighted[0]):
                dataline = "{0} {1} {2} {3} {4} {5} {6} {7}\n".format(
                    ("%.3f" % (pred_i.centroid_weighted[1])).zfill(8), ("%.3f" % (pred_i.centroid_weighted[0])).zfill(8),
                    "%04d" % pred_i.area,
                    "%04d" % (pred_i.bbox[3] - pred_i.bbox[1]), "%04d" % (pred_i.bbox[2] - pred_i.bbox[0]),
                    "0000000000",
                    "00000.000", "00000.000"
                )
                ipd_file.write(dataline)
        ipd_file.close()

        df_pred = pd.read_table(ipd_fname, sep='\s+', header=None, encoding='utf-8',
                                names=['col', 'row', 'pixnum', 'length', 'width', 'graysum', 'bgavg', 'bgvar'])
        # print(time.time()-a)
        fname = os.path.join('dataset', dataset, 'ipd',filename.replace('.png','.IPD'))
        df_gt = pd.read_table(fname, sep='\s+', header=None, encoding='utf-8',
                                names=['col', 'row', 'pixnum', 'length', 'width', 'graysum', 'bgavg',
                                        'bgvar'])

        ''' use KD-Tree '''
        thres = 1.5
        all_pred = len(df_pred)
        all_gt = len(df_gt)

        kd_gt = KDTree(df_gt[['col', 'row']].values)
        match = kd_gt.query(df_pred[['col', 'row']].values, k=1)
        tp_idx_fromgt = np.unique(match[1][match[0] < thres])
        tp_objs = df_gt.copy().iloc[tp_idx_fromgt]
        fn_objs = df_gt.copy().drop(labels=tp_idx_fromgt)
        # static snr
        for _, tp_item in tp_objs.iterrows():
            #snr_int = int(tp_item['snr'])
            snr_int = 1
            if snr_int < 0:
                print('a minus snr!')
            elif snr_int <= bins:
                tp_count_array[snr_int] += 1
            elif snr_int > bins:
                tp_count_array[bins] += 1
        for _, fn_item in fn_objs.iterrows():
            # snr_int = int(fn_item['snr'])
            snr_int = 1
            if snr_int < 0:
                print('a minus snr!')
            elif snr_int <= bins:
                fn_count_array[snr_int] += 1
            elif snr_int > bins:
                fn_count_array[bins] += 1

        tp = len(np.unique(match[1][match[0] < thres]))
        fp = all_pred - tp
        fn = all_gt - tp
        if tp + fp == 0:
            precision = 0
        else:
            precision = tp / (tp + fp)
        if tp + fn==0:
            recall=0
        else:
            recall = tp / (tp + fn)
        if precision != 0 and recall != 0:
            f1 = 2 * precision * recall / (precision + recall)
            faRate = (1 / precision - 1) * recall
        else:
            f1 = 0
            faRate = 0
        count += 1
        pre_sum += precision
        rec_sum += recall
        f1_sum += f1
        fa_sum += faRate

    print('tp-snr: {}'.format(tp_count_array))
    print('fn-snr: {}'.format(fn_count_array))

    precision = pre_sum / count
    recall = rec_sum / count
    f1 = f1_sum / count
    faRate = fa_sum / count
    print('precision:{}, recall:{}, f1:{}, faRate:{}'.format(precision, recall, f1, faRate))