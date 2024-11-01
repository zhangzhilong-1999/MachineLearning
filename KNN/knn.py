import numpy as np
from tqdm import tqdm 

class KnnClassifier:
    def __init__(self, k=1):
        self.k = k
        
    def k_nearest(self, X, y, x):
        if X is None:
            return 
        distance_point_label = [[np.linalg.norm(x-Xi), Xi, yi] for Xi,yi in zip(X,y)]
        distance_point_label.sort(key= lambda x: x[0]) # 按第一个位置的值升序排列
        best_k = distance_point_label[0:self.k] # 左闭右开
        return best_k

    def _predict(self, X, y, x):
        best_k_label = [best[2] for best in self.k_nearest(X, y, x)]
        counter_dict = {}
        for label in best_k_label:
            counter_dict.setdefault(label,0)
            counter_dict[label] += 1
        counter_dict = sorted(counter_dict.items(), key = lambda x: x[1])
        return counter_dict[-1][0]
    
    def predict(self, X, y, equery):
        return np.array([self._predict(X, y, x) for x in tqdm(equery)])