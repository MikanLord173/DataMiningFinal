import numpy as np

class KMeans:
    # 對每筆元素（重新）分群，回傳新分群結果每群的中心點
    def __init__(self, k, max_iters=100, random_state=57):
        self.k = k
        self.max_iters = max_iters
        self.random_state = random_state

    # kmeans++ 初始化
    def _pp_init(self, data):
        np.random.seed(self.random_state)
        # n_data 紀錄總共有幾筆資料，若 data 為 i*j 矩陣，data.shape 回傳 (i, j)
        n_data = data.shape[0]
        centers = []

        # 隨機抽取一點作為第一個中心
        first_center_index = np.random.randint(n_data)
        centers.append(data[first_center_index])

        # 重複 k-1 次
        for _ in range(1, self.k):
            distances_list = []
            for x in data:
                distances_list.append(min(np.linalg.norm(x - c)**2 for c in centers))
            distances = np.array(distances_list)

            probs = distances / distances.sum()
            new_center_index = np.random.choice(n_data, p=probs)
            centers.append(data[new_center_index])

        return np.array(centers)

    def _cluster(self, centers, data):
        # data中每個點跟所有中心點的距離
        # 例：k=3, data entries=3
        # distances = 
        # [[2.0, 1.2, 1.7],
        #  [5.7, 0.5, 5.7],
        #  [2.1, 7.1, 7.3]]
        distances = np.linalg.norm(data[:, np.newaxis] - centers, axis=2)

        # 距離data中每個點最近的中心點索引（0 ~ k-1）
        # 以上面註解的 distances 為例：
        # nearest_center_indices = [1, 1, 0]
        nearest_center_indices = np.argmin(distances, axis=1)

        new_centers = []
        for center_index in range(self.k):
            if np.any(nearest_center_indices == center_index):
                # 從第 i 群中找出新中心點
                new_centers.append(data[nearest_center_indices == center_index].mean(axis=0))
            else:
                # 若第 i 群沒有資料，沿用舊的
                new_centers.append(centers[center_index])
        return np.array(new_centers), nearest_center_indices

    # 回傳每筆資料對應的群組索引
    # pp: 是否改用 kmeans++
    def predict(self, data, pp=False):
        if pp:
            initial_centers = self._pp_init(data)
        else:
            np.random.seed(self.random_state)
            random_k_indices = np.random.choice(len(data), self.k, replace=False) # 在data的長度內隨機選取k個索引，replace代表是否可選重複數值
            initial_centers = data[random_k_indices] # 紀錄隨機的k個中心點

        centers = initial_centers
        for _ in range(self.max_iters):
            new_centers, indices = self._cluster(centers, data)

            if np.allclose(centers, new_centers):
                break
            centers = new_centers
        return indices
