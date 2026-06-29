import os
import shutil
from tqdm import tqdm

def copy_files_and_generate_txt(src_dir, dst_dir, txt_file_path):
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)
    
    filenames = []
    src_pic_dir=os.path.join(src_dir,'images')
    src_ipd_dir=os.path.join(src_dir,'ipd_all')
    for filename in tqdm(os.listdir(src_pic_dir)):
        src_file_path = os.path.join(src_pic_dir, filename)
        src_ipd_path = os.path.join(src_ipd_dir, os.path.splitext(filename)[0]+'.IPD')
        if os.path.isfile(src_file_path):
            dst_file_path = os.path.join(dst_dir, 'fits',filename)
            shutil.copy2(src_file_path, dst_file_path)
            dst_ipd_path = os.path.join(dst_dir, 'ipd', os.path.splitext(filename)[0]+'.IPD')
            shutil.copy2(src_ipd_path, dst_ipd_path)
            filenames.append(os.path.splitext(filename)[0])
    
    with open(txt_file_path, 'w') as txt_file:
        for name in filenames:
            txt_file.write(name + '\n')

# src_dir1 = '/home/cjz/project/cdnkeydet/cdnkeydet/data/Sat/realdata_ast/train1024/'
# dst_dir1 = '/home/lixy/workspace/cdnnet/cdnnet-231010/cdnnet/dataset/new_data_ast'
# txt_file_path1 = '/home/lixy/workspace/cdnnet/cdnnet-231010/cdnnet/dataset/new_data_ast/train_test/train.txt'
src_dir1 = '/media/zhige/data/bh_target_detection/xlj_change/dataset/fitstrans/fitsdata/5/train/'
dst_dir1 = '/media/zhige/data/bh_target_detection/xlj_change/dataset/fits_softCircle_modified/train_test1/'
txt_file_path1 = '/media/zhige/data/bh_target_detection/xlj_change/dataset/fits_softCircle_modified/train_test1/train.txt'
copy_files_and_generate_txt(src_dir1, dst_dir1, txt_file_path1)

src_dir2 = '/media/zhige/data/bh_target_detection/xlj_change/dataset/fitstrans/fitsdata/5/test/'
dst_dir2 = '/media/zhige/data/bh_target_detection/xlj_change/dataset/fits_softCircle_modified/train_test1/'
txt_file_path2 = '/media/zhige/data/bh_target_detection/xlj_change/dataset/fits_softCircle_modified/train_test1/test.txt'
copy_files_and_generate_txt(src_dir2, dst_dir2, txt_file_path2)
