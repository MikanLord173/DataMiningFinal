import os
import pandas as pd
import numpy as np
import process
from plot import plot2d, plot2d_subplots

from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression

data_set_names = ("Arrhythmia Data Set", "gene expression cancer RNA-Seq Data Set")
using_data_set = 0

current_dir = os.getcwd()
dir_str = current_dir + "/DataMining/final/"

train_data = pd.read_csv(dir_str + f"{data_set_names[using_data_set]}/train_data.csv", header=None).values
train_label = pd.read_csv(dir_str + f"{data_set_names[using_data_set]}/train_label.csv", header=None)[0]
test_data = pd.read_csv(dir_str + f"{data_set_names[using_data_set]}/test_data.csv", header=None).values
test_label = pd.read_csv(dir_str + f"{data_set_names[using_data_set]}/test_label.csv", header=None)[0].values

# 補上缺失值
train_data_imputed, test_data_imputed = process.impute(train_data=train_data, test_data=test_data, strategy='mean')
# 標準化
train_data_standardized, test_data_standardized = process.standardize(train_data=train_data_imputed, test_data=test_data_imputed)
# 挑出離群值
train_data_IQR, train_label_IQR = process.IQR(train_data=train_data_standardized, train_label=train_label, target=(1,))
# 超取樣
train_data_processed, train_label_processed = process.oversampling(train_data=train_data_IQR, train_label=train_label_IQR)

datasets = [train_data_imputed, train_data_standardized, train_data_IQR, train_data_processed]
labels = [train_label, train_label, train_label_IQR, train_label_processed]
titles = ["Imputed", "Standardlized", "After IQR", "Oversampled"]

plot2d_subplots(datasets, labels, titles)

clfs = (
    RandomForestClassifier(n_estimators=100),   # 隨機森林
    KNeighborsClassifier(n_neighbors=20),       # KNN
    GaussianNB(),                               # Naive Bayes
    LogisticRegression()                        # 邏輯回歸
)
clf_index = 0

clfs[clf_index].fit(train_data, train_label)
pred_label = clfs[clf_index].predict(test_data)
pred_probas = clfs[clf_index].predict_proba(test_data)
max_proba = np.max(pred_probas, axis=1)

adj_pred_label = np.where(max_proba >= 0.5, pred_label, -1)

df = pd.DataFrame({
    'True Label': test_label,
    'Pred Label': pred_label,
    'Adjusted Label': adj_pred_label,
    'Probability': max_proba
})

df.to_excel(dir_str + f"test{clf_index}.xlsx", index=True)

unsure_data = test_data[max_proba < 0.5]
unsure_label_true = test_label[max_proba < 0.5]

df_unsure = pd.DataFrame(unsure_data)
df_unsure['True Label'] = unsure_label_true

df_unsure.to_excel(dir_str + "unsure.xlsx", index=True)

'''kmeans = KMeans(n_clusters=5, random_state=0)
clusters = kmeans.fit_predict(unsure_data)

df_unsure['Cluster'] = clusters

df_unsure.to_excel(dir_str + "unsure_clustered.xlsx", index=True)'''