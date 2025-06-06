import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix

def plot2d(dataset, label, title:str, path:str=None):
    pca = PCA(n_components=2)   # 建立一個 PCA 降維器，目標降到 2 維
    data_2d = pca.fit_transform(dataset) # 將高維特徵 X 降成 2 維資料 X_2d

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(data_2d[:, 0], data_2d[:, 1], c=label, cmap='tab10', alpha=0.7)
    # 畫出散點圖：
    # - X[:, 0] 是 x 座標（第一維特徵）
    # - X[:, 1] 是 y 座標（第二維特徵）
    # - c=y：根據 label 上色
    # - cmap='tab10'：使用 10 色分類色盤
    # - alpha=0.7：設定透明度，讓圖點不會太實心
    plt.title(title)

    # 根據 label 自動產生圖例（legend），每個顏色對應一個類別
    legend = plt.legend(*scatter.legend_elements(), title="Labels")

    plt.gca().add_artist(legend)    # 把圖例加到目前的座標軸中
    plt.grid(True)
    if path:
        plt.savefig(path)
    plt.show()

def plot2d_subplots(datasets, labels, titles, size=(2, 2), path:str=None):
    """
    datasets: List of data arrays (每筆資料 shape 都應為 [n_samples, n_features])
    labels_list: 對應的 label 或群組 list，每個 element 是對應一個資料集的 label
    titles: 每張子圖的標題 list
    """

    fig, axes = plt.subplots(size[0], size[1], figsize=(12, 10))
    axes = axes.flatten()

    for i, (X, y) in enumerate(zip(datasets, labels)):

        pca = PCA(n_components=2)
        X_2d = pca.fit_transform(X)
        ax = axes[i]

        for label in np.unique(y):
            mask = y == label
            count = np.sum(y == label)
            ax.scatter(X_2d[mask, 0], X_2d[mask, 1], label=f'{label} ({count})', alpha=0.7)

        ax.set_title(titles[i])
        ax.legend()
        ax.grid(True)

    plt.tight_layout()
    if path:
        plt.savefig(path)
    plt.show()

# 繪製混淆矩陣
def plot_cm(label1, label2, name1, name2, path:str=None):
    all_labels = []
    for i in range(1, 14):
        all_labels.append(f'Class {i}')
    for i in range(5):
        all_labels.append(f'Cluster {i}')
    cm = confusion_matrix(label1, label2, labels=all_labels)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=all_labels, yticklabels=all_labels)
    plt.xlabel(name2)
    plt.ylabel(name1)
    plt.title('Confusion Matrix')
    plt.tight_layout()
    if path:
        plt.savefig(path)
    plt.show()