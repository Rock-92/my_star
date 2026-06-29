import os.path

import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'serif'
plt.rcParams["font.family"] = ["Times New Roman"] + plt.rcParams['font.serif']
from matplotlib.pyplot import MultipleLocator
import numpy as np

def getCurveData(fpath, epoch_range):
    curve_pre = []
    curve_rec = []
    with open(fpath) as f:
        while True:
            line = f.readline()
            if line.strip():
                line_split = line.split()
                pre = line_split[1].split(':')[-1].replace(',', '')
                rec = line_split[2].split(':')[-1].replace(',', '')
                curve_pre.append(float(pre))
                curve_rec.append(float(rec))
            else:
                break
    curve_pre = np.array(curve_pre)[epoch_range]
    curve_rec = np.array(curve_rec)[epoch_range]
    return curve_pre, curve_rec

def drawAndSave(fpath_list, epoch_range, label_list, outpath):
    curve_num = len(fpath_list)
    curve_pre_list = [None] * curve_num
    curve_rec_list = [None] * curve_num
    color_list = ['pink', 'blue', 'green', 'yellow', 'cyan', 'magenta', 'red']
    for i in range(curve_num):
        fpath = fpath_list[i]
        curve_pre_list[i], curve_rec_list[i] = getCurveData(fpath, epoch_range)

    plt.figure()
    for i in range(curve_num):
        if i==0:
            plt.plot(epoch_range, curve_rec_list[i], color=color_list[i], label="{} precision".format(label_list[i]))
        else:
            plt.plot(epoch_range, curve_rec_list[i], color=color_list[i], label="{} precision".format(label_list[i]))
    # l1 = plt.legend(loc='lower right')
    # handle_list = [None] * curve_num
    # for i in range(curve_num):
    #     if i==0:
    #         handle_list[i], = plt.plot(epoch_range, curve_rec_list[i], color=color_list[i],
    #                                   label="{}".format(label_list[i]))
    #     else:
    #         handle_list[i], = plt.plot(epoch_range, curve_rec_list[i], color=color_list[i], linestyle='dashed',
    #                                   label="{}".format(label_list[i]))
    # plt.legend(handles=handle_list, loc='upper right', scatterpoints=1)
    plt.legend(loc='lower right', scatterpoints=1)
    plt.xlabel('epoch', fontsize=12)
    plt.ylabel('detection rate', fontsize=12)
    # plt.gca().add_artist(l1)
    x_major_locator = MultipleLocator(5)
    ax = plt.gca()
    ax.xaxis.set_major_locator(x_major_locator)
    plt.xlim(epoch_range.min(), epoch_range.max())
    plt.grid()
    plt.savefig(outpath, bbox_inches='tight', pad_inches=0.1)


if __name__ == '__main__':

    ''' 1. wo deconv '''
    method_name = 'cdn_b_rfb-cdn_b_resrfb'
    fpath_list = [os.path.join(
                      '/media/zhige/data/bh_target_detection/xlj_change/result/01999_DNRFANet_block_RFB/DNRFANet_fits_softCircle_modified_good_precision_recall.log'
                  ),
                  os.path.join(
                      '/media/zhige/data/bh_target_detection/xlj_change/result/01999_DNRFANet_block_ResRFB/DNRFANet_fits_softCircle_modified_good_precision_recall.log'
                  )
                  ]
    epoch_range = np.arange(70, 100)
    label_list = ['DNRFANet_b_RFB', 'DNRFANet_b_ResRFB']
    outpath = '../outfiles/plotDetMetricCurve/wo{}.png'.format(method_name)
    drawAndSave(fpath_list, epoch_range, label_list, outpath)

    ''' 2. wo hierarchy '''
    method_name = 'hierarchy'
    fpath_list = [os.path.join(
                      '/media/zhige/data/bh_target_detection/xlj_change/result/fits_softCircle_modified_DNRFANet_20241026_16:01:00/DNRFANet_fits_softCircle_modified_good_precision_recall.log'
                  ),
                  os.path.join(
                      '/media/zhige/data/bh_target_detection/xlj_change/result/fits_softCircle_modified_CDNRFBNet_20241027_x0_0only/CDNRFBNet_fits_softCircle_modified_good_precision_recall.log'
                  )
                  ]
    epoch_range = np.arange(70, 100)
    label_list = ['DNRFANet', 'wo/{}'.format(method_name)]
    outpath = '../outfiles/plotDetMetricCurve/wo{}.png'.format(method_name)
    drawAndSave(fpath_list, epoch_range, label_list, outpath)

    ''' 3. wo softmask '''
    method_name = 'softmask'
    fpath_list = [os.path.join(
                      '/media/zhige/data/bh_target_detection/xlj_change/result/fits_softCircle_modified_DNRFANet_20241026_16:01:00/DNRFANet_fits_softCircle_modified_training_loss.log'
                  ),
                  os.path.join(
                      '/media/zhige/data/bh_target_detection/xlj_change/result/fits_softCircle_modified_CDNRFBNet_20241027_x0_0only/CDNRFBNet_fits_softCircle_modified_training_loss.log'
                  )
                  ]
    epoch_range = np.arange(70, 100)
    label_list = ['DNRFANet', 'CDNRFB-Net']
    outpath = '../outfiles/plotDetMetricCurve/wo{}.png'.format(method_name)
    drawAndSave(fpath_list, epoch_range, label_list, outpath)

    ''' 4. all method recall '''
    method_name = 'all method recall'
    fpath_list = [os.path.join(
        'D:/DNRFA-Net/result/fits_softCircle_modified_DNRFANet_20250223/DNRFANet_fits_softCircle_modified_good_precision_recall.log'
    ),
        os.path.join(
            'D:/DNRFA-Net/result/fits_softCircle_modified_DNANet_20250227/DNANet_fits_softCircle_modified_good_precision_recall.log'
        ),
        os.path.join(
            'D:/DNRFA-Net/result/fits_softCircle_modified_UNet_20250225/UNet_fits_softCircle_modified_good_precision_recall.log'
        ),
        os.path.join(
            'D:/DNRFA-Net/result/fits_softCircle_modified_AGPCNet_20250225/AGPCNet_fits_softCircle_modified_good_precision_recall.log'
        ),
        os.path.join(
            'D:/DNRFA-Net/result/fits_softCircle_modified_HCFNet_20250226/HCFNet_fits_softCircle_modified_good_precision_recall.log'
        ),
        os.path.join(
            'D:/DNRFA-Net/result/fits_softCircle_modified_SeRankDet_20250227/SeRankDet_fits_softCircle_modified_good_precision_recall.log'
        )
    ]
    epoch_range = np.arange(70, 100)
    label_list = ['DNRFANet', 'DNANet', 'UNet', 'AGPCNet', 'HCFNet', 'SeRankDet']
    outpath = 'D:/DNRFA-Net/outfiles/plotDetMetricCurve/{}.png'.format(method_name)
    drawAndSave(fpath_list, epoch_range, label_list, outpath)