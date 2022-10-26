import pandas as pd
import numpy as np

train_path = '/home/hanrui/train_test_data_all/test.csv'
test_path = '/home/hanrui/train_test_data_all/test.csv'
label_path = '/home/hanrui/train_test_data_all/label_dict.npy'

# model = "hfl/chinese-macbert-base"
model = '/mnt/hdd3/hanrui/bert/result'


def reverse_labels(label_path):  # eg: {'急性上呼吸道感染': 0, ...}
    label_dict = np.load(label_path, allow_pickle=True).tolist()
    label_dict_reverse = {}
    for key, val in label_dict.items():
        label_dict_reverse[val] = int(key)
    return label_dict_reverse


def preprocess_data(path):  # 修改index为sentence，label
    df = pd.read_csv(path)
    df.rename(columns={'content': 'sentence'}, inplace=True)
    return df


if __name__ == '__main__':
    df = preprocess_data(test_path)
    print(df)
