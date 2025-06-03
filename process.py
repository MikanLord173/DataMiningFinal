# 此檔案中的函數用於對資料進行前處理

import pandas as pd
from sklearn.impute import SimpleImputer    # 補上缺失值
from sklearn.preprocessing import StandardScaler    # 標準化
from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import SMOTE

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

def oversampling(train_data, train_label):
    #smote = SMOTE(random_state=57, k_neighbors=1)
    #return smote.fit_resample(train_data, train_label)
    ros = RandomOverSampler(random_state=57)
    return ros.fit_resample(train_data, train_label)