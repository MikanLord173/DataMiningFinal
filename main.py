import os
import pandas as pd
import numpy as np
import process
import matplotlib.pyplot as plt
from plot import plot2d, plot2d_subplots, plot_cm

from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression

from kmeans import KMeans

## ------------------- 可調整參數 ------------------- ##

# 0: 補缺失值 -> 挑出離群值 (IQR) -> 標準化 -> oversampling
# 1: 補缺失值 -> 標準化 -> 挑出離群值 (zscore) -> oversampling
# 2: 補缺失值 -> 標準化 -> undersampling -> oversampling
METHOD = 0
# 0: Random Forest
# 1: KNN
# 2: Naive Bayes
# 3: 邏輯回歸
CLF = 2
# 開關 K-means++
PP = False
# 分類信心低於 UNKNOWN_THRESHOLD 會被分成未知類（之後拿去分群）
UNKNOWN_THRESHOLD = 0.4

## ------------------------------------------------ ##

# 改一個變數即變更前處理的方法
method_func = (process.method0, process.method1, process.method2)
subplot_titles = (
    ["Imputed", "Outliers removed (IQR)", "Standardlized", "Oversampled"],
    ["Imputed", "Standardlized", "Outliers removed (zscore)", "Oversampled"],
    ["Imputed", "Standardlized", "Undersampled", "Oversampled"]
)

# 把分類器放到 tuple 裡，這樣只要改一個變數就能更換使用的分類器
clfs = (
    RandomForestClassifier(n_estimators=217),   # 隨機森林
    KNeighborsClassifier(n_neighbors=20),       # KNN
    GaussianNB(),                               # Naive Bayes
    LogisticRegression()                        # 邏輯回歸
)

# 用來挑選資料集
data_set_names = ("Arrhythmia Data Set", "gene expression cancer RNA-Seq Data Set")
using_data_set = 0

# 取得當前路徑，以便讀取/寫入檔案
current_dir = os.getcwd()
dir_str = current_dir + "/DataMining/final/"
os.makedirs(f"{dir_str}/Figures/method{METHOD}_clf{CLF}_{"no" if not PP else ""}pp", exist_ok=True)

# 從 csv 檔讀取各項資料
# 資料型態：data -> np.ndarray; label -> pd.Series
train_data_init = pd.read_csv(dir_str + f"{data_set_names[using_data_set]}/train_data.csv", header=None).values
train_label_init = pd.read_csv(dir_str + f"{data_set_names[using_data_set]}/train_label.csv", header=None)[0]
test_data_init = pd.read_csv(dir_str + f"{data_set_names[using_data_set]}/test_data.csv", header=None).values
test_label = pd.read_csv(dir_str + f"{data_set_names[using_data_set]}/test_label.csv", header=None)[0].values

train_data, test_data, train_label, datasets, labels = method_func[METHOD](train_data_init, test_data_init, train_label_init)
plot2d_subplots(datasets, labels, subplot_titles[METHOD], size=(2, 2), path=f"{dir_str}/Figures/method{METHOD}_clf{CLF}_{"no" if not PP else ""}pp/4steps.png") # 四個步驟的結果分別製作一張分佈圖

clfs[CLF].fit(train_data, train_label)    # 訓練資料放入模型
pred_label = clfs[CLF].predict(test_data) # 預測結果的標籤 (type: ndarray)
pred_probas = clfs[CLF].predict_proba(test_data)  # 每個標籤的可能機率 (type: ndarray)
max_proba = np.max(pred_probas, axis=1) # 可能機率中最高的那個（也就是預測結果的機率）(type: ndarray)

# 若最高機率低於UNKNOWN_THRESHOLD，把標籤改成 "Unknown" 然後存進 adj_pred_label (type: ndarray)
adjusted_label = np.where(max_proba >= UNKNOWN_THRESHOLD, pred_label, "Unknown") 

# 把上述資訊整理到 DataFrame 以便輸出到 excel 觀察
df = pd.DataFrame({
    'True Label': test_label,
    'Pred Label': pred_label,
    'Adjusted Label': adjusted_label,
    'Probability': max_proba
})

df.to_excel(dir_str + f"test{CLF}.xlsx", index=True)

# 製圖，上半為預處理後的訓練資料；下半為測試資料
datasets = [train_data, test_data]
labels = [train_label, adjusted_label]
titles = ["Train data", "Classified test data"]
plot2d_subplots(datasets, labels, titles, size=(2, 1), path=f"{dir_str}/Figures/method{METHOD}_clf{CLF}_{"no" if not PP else ""}pp/Train_Test.png")

unsure_mask = adjusted_label == "Unknown"
unsure_data = test_data[unsure_mask]

kmeans = KMeans(k=5)
clusters = kmeans.predict(data=unsure_data, pp=PP)
final_label = adjusted_label.copy()
final_label[unsure_mask] = [f"Cluster {i}" for i in clusters]

# 製圖，上半為分類但未分群的資料，下半為分群後的資料
datasets = [test_data, test_data]
labels = [adjusted_label, final_label]
titles = ["Classified", "Classified + Clustered"]
plot2d_subplots(datasets, labels, titles, size=(2, 1), path=f"{dir_str}/Figures/method{METHOD}_clf{CLF}_{"no" if not PP else ""}pp/Classified_Clustered.png")

test_label = test_label.astype(str)
for i in range(1, 14):
    test_label[test_label == str(i)] = f'Class {i}'
    final_label[final_label == str(i)] = f'Class {i}'

# 繪製混淆矩陣
plot_cm(test_label, final_label, "True label", "Predicted label", path=f"{dir_str}/Figures/method{METHOD}_clf{CLF}_{"no" if not PP else ""}pp/ConfusionMatrix.png")
print(f'Accuracy: {np.mean(test_label == final_label)}')