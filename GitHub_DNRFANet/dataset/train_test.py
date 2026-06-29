import os


def generate_dataset_labels(folder_path, test_suffix="01999", output_dir="."):

    all_files = []
    for filename in os.listdir(folder_path):
        if filename.endswith('.fits'):
            base_name = filename[:-5]
            all_files.append(base_name)

    train_files = []
    test_files = []

    for file_name in all_files:
        parts = file_name.split('_')
        if len(parts) >= 2:
            if test_suffix in parts[0]:
                test_files.append(file_name)
            else:
                train_files.append(file_name)
        else:
            train_files.append(file_name)

    train_files.sort()
    test_files.sort()

    train_path = 'D:/DNRFA-Net/dataset/fits_softCircle_modified/8192_256/train.txt'
    with open(train_path, 'w') as f:
        for file_name in train_files:
            f.write(file_name + '\n')

    test_path = 'D:/DNRFA-Net/dataset/fits_softCircle_modified/8192_256/test.txt'
    with open(test_path, 'w') as f:
        for file_name in test_files:
            f.write(file_name + '\n')

    print(f"train.txt save to: {train_path}")
    print(f"test.txt save to: {test_path}")


# 使用示例
if __name__ == "__main__":
    fits_folder = "D:/DNRFA-Net/dataset/fits_softCircle_modified/8192_fits/"

    generate_dataset_labels(fits_folder)