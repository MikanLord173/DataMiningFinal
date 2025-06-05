# 此檔案中的函數用於對資料進行前處理

import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer    # 補上缺失值
from sklearn.preprocessing import StandardScaler    # 標準化
from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import SMOTE
from scipy.stats import zscore

# 補缺失值 -> 挑出離群值 (IQR) -> 標準化 -> oversampling
def method0(train_data, test_data, train_label):
    datasets = []
    labels = []

    # 補上缺失值
    train_data, test_data = impute(train_data=train_data, test_data=test_data, strategy='mean')
    datasets.append(train_data)
    labels.append(train_label)
    # 挑出離群值 (IQR)
    train_data, train_label = IQR(train_data=train_data, train_label=train_label, target=(1,))
    datasets.append(train_data)
    labels.append(train_label)
    # 標準化
    train_data, test_data = standardize(train_data=train_data, test_data=test_data)
    datasets.append(train_data)
    labels.append(train_label)
    # 超取樣
    train_data, train_label = oversampling(train_data=train_data, train_label=train_label, max_count=35)
    datasets.append(train_data)
    labels.append(train_label)

    return train_data, test_data, train_label, datasets, labels

# 補缺失值 -> 標準化 -> 挑出離群值 (zscore) -> oversampling
def method1(train_data, test_data, train_label):
    datasets = []
    labels = []

    # 補上缺失值
    train_data, test_data = impute(train_data=train_data, test_data=test_data, strategy='mean')
    datasets.append(train_data)
    labels.append(train_label)
    # 標準化
    train_data, test_data = standardize(train_data=train_data, test_data=test_data)
    datasets.append(train_data)
    labels.append(train_label)
    # 挑出離群值 (zscore)
    train_data, train_label = zscore_outliers(train_data=train_data, train_label=train_label, threshold=3, target=(1,))
    datasets.append(train_data)
    labels.append(train_label)
    # 超取樣
    train_data, train_label = oversampling(train_data=train_data, train_label=train_label, max_count=35)
    datasets.append(train_data)
    labels.append(train_label)

    return train_data, test_data, train_label, datasets, labels

# 補缺失值 -> 標準化 -> undersampling -> oversampling
def method2(train_data, test_data, train_label):
    datasets = []
    labels = []

    # 補上缺失值
    train_data, test_data = impute(train_data=train_data, test_data=test_data, strategy='mean')
    datasets.append(train_data)
    labels.append(train_label)
    # 標準化
    train_data, test_data = standardize(train_data=train_data, test_data=test_data)
    datasets.append(train_data)
    labels.append(train_label)
    # 欠採樣
    train_data, train_label = undersampling(train_data=train_data, train_label=train_label, max_count=50)
    datasets.append(train_data)
    labels.append(train_label)
    # 超取樣
    train_data, train_label = oversampling(train_data=train_data, train_label=train_label, max_count=35)
    datasets.append(train_data)
    labels.append(train_label)

    return train_data, test_data, train_label, datasets, labels

# 補上資料中缺失的欄位
def impute(train_data, test_data, strategy='mean'):
    imp = SimpleImputer(strategy=strategy) # 用於補上資料中缺失的值，strategy 亦可選 'median', 'most_frequent', 'constant'
    train_data = imp.fit_transform(train_data)
    test_data = imp.transform(test_data)
    return train_data, test_data

# 將資料標準化
def standardize(train_data, test_data):
    scaler = StandardScaler()
    train_data = scaler.fit_transform(train_data)
    test_data = scaler.transform(test_data) # 訓練、測試都要用相同的標準來標準化，所以不能再 fit_transform()
    return train_data, test_data

# 取得 df 中離群值的位置，是裝有 True / False 的 pd.Series
def get_mask(df):
    Q1 = df.quantile(0.25)
    Q3 = df.quantile(0.75)
    IQR = Q3 - Q1
    # 回傳離群值的資料
    return ((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))).any(axis=1)

def zscore_outliers(train_data, train_label, threshold, target:tuple=None):
    data_df = pd.DataFrame(train_data)
    label_series = pd.Series(train_label)

    # 若有限定要處理的標籤
    if target:
        # 先複製一個 DataFrame，並加上對應的標籤
        full_df = pd.DataFrame(train_data)
        full_df['label'] = train_label

        # 找出包含對應標籤的資料
        target_rows = full_df['label'].isin(target)

        # 在對應標籤的資料中找出離群值位置
        zscores = zscore(train_data[target_rows])
        mask = (np.abs(zscores) > threshold).any(axis=1)

        # 轉成 full_df 的索引形式（mask 是從對應標籤的資料中取出來的，索引會和 full_df 不對齊）
        indices = full_df[target_rows].loc[mask].index

        # 根據索引挑出對應的資料 / 標籤
        cleaned_data = data_df.drop(index=indices).values
        cleaned_label = label_series.drop(index=indices).values
    else:
        zscores = zscore(train_data)
        mask = (np.abs(zscores) > threshold).any(axis=1)
        cleaned_data = data_df[~mask].values
        cleaned_label = label_series[~mask].values
    return cleaned_data, cleaned_label

# 用IQR方法移除訓練資料的離群值，只處理 target 中有的 label（若None則全部處理）
def IQR(train_data, train_label, target:tuple=None):
    data_df = pd.DataFrame(train_data)
    label_series = pd.Series(train_label)

    # 若有限定要處理的標籤
    if target:
        # 先複製一個 DataFrame，並加上對應的標籤
        full_df = pd.DataFrame(train_data)
        full_df['label'] = train_label

        # 找出包含對應標籤的資料
        target_rows = full_df['label'].isin(target)

        # 在對應標籤的資料中找出離群值位置
        mask = get_mask(full_df.loc[target_rows].drop(columns='label'))

        # 轉成 full_df 的索引形式（mask 是從對應標籤的資料中取出來的，索引會和 full_df 不對齊）
        indices = full_df[target_rows].loc[mask].index

        # 根據索引挑出對應的資料 / 標籤
        cleaned_data = data_df.drop(index=indices).values
        cleaned_label = label_series.drop(index=indices).values
    else:
        mask = get_mask(data_df)
        cleaned_data = data_df[~mask].values
        cleaned_label = label_series[~mask].values
    return cleaned_data, cleaned_label

# 讓所有標籤的樣本數控制在 max_count 以下
def undersampling(train_data, train_label, max_count:int):
    final_indices = []

    for lbl in np.unique(train_label):
        indices = np.where(train_label == lbl)[0] # 回傳的為 tuple: (array([  位置a, 位置b, 位置c, ...]),)，所以要用 [0] 取出
        if len(indices) > max_count:
            sampled = np.random.choice(indices, max_count, replace=False) # replace 代表是否可選重複值
        else:
            sampled = indices
        final_indices.extend(sampled) # 跟 append() 不同，extend() 不會將整個 array 視為一整個物件加進去

    return train_data[final_indices], train_label[final_indices]

def oversampling(train_data, train_label, max_count=None):
    label_count = dict()
    for label in np.unique(train_label):
        label_count[label] = np.sum(train_label == label)
    if not max_count:
        max_count = max(label_count.values())
    strategy = dict()
    for key, value in label_count.items():
        if value >= 5:
            strategy[key] = max(max_count, value)

    ros = RandomOverSampler(random_state=57, sampling_strategy=strategy)
    data_resampled, label_resampled = ros.fit_resample(train_data, train_label)
    return data_resampled, label_resampled