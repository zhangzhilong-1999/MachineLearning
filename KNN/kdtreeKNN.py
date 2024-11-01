# 自己写出knn的kd树实现
import numpy as np
from tqdm import tqdm
from numpy.linalg import norm

class KdNode:
    def __init__(self, point = None, label = None, fi = None, fv = None, left = None, right = None):
        self.point = point
        self.label = label
        self.fi = fi
        self.fv = fv
        self.left = left
        self.right = right

class KdTreeKNN:
    def __init__(self, k=1):
        ''' 
        :param k k值
        :param tree kd树
        '''
        self.k = k
        self.tree = None

    def Build(self, X, y, depth = 0):
        ''' 
        :param X        train set
        :param y        label
        :param depth    depth
        '''
        # 递归终止条件
        if X is None:       # 如果X是空集, not X =1,执行下面的命令.    也可以用 if X is None
            return 
        n_samples, n_features = X.shape
        if n_samples == 1:
            return KdNode(point = X[0], label = y[0])

        fi = depth % n_features
         # 按选定的fi坐标轴取出元素
        argsort = np.argsort(X[:,fi])
        idx_middle = argsort[n_samples // 2]
        idxs_left = argsort[:n_samples // 2] # 左闭右开
        idxs_right = argsort[n_samples // 2+1:]
        fv = X[idx_middle, fi]

        left, right = None, None
        if len(idxs_left) > 0:
            left = self.Build(X[idxs_left], y[idxs_left], depth + 1)
        if len(idxs_right) > 0:
            right = self.Build(X[idxs_right], y[idxs_right],  depth + 1)
        return KdNode(X[idx_middle], y[idx_middle], fi, fv, left, right)
    
    def fit(self, X, y):
        ''' 
        :param X        train set
        :param y        label
        '''
        self.tree = self.Build(X, y)

    def nearest(self, kdtree, x, k=1, best_k= None):
        '''
        :param kdtree 给定数据集(X,y)生成的kd树
        :param x 要预测的输入数据
        :param k    k值
        :param best_k 要从上次递归中接受, k个
        在kdtree中找到equery的k近邻点
        ''' 
        # 递归终止条件， 放在递归调用之前
        if not kdtree:
            return
        
        tem_distance = norm(x - kdtree.point)
        if best_k is None:
            '''
            best_distance = norm(x - kdtree.point)
            best_point = kdtree.point
            best = [best_distance, best_point]
            best_k.append(best)
            '''
            best_k= [[tem_distance, kdtree.point, kdtree.label]] * k # 凑够k个近邻点
        elif tem_distance < best_k[-1][0]:
            best_k[-1] = [tem_distance, kdtree.point, kdtree.label] # 凑够之后，替换
            best_k.sort(key= lambda x: x[0], reverse= False)  # 按元素的第一个分量升序排列 

        if kdtree.left is None and kdtree.right is None:    # and
            return best_k

        # 在kd树中找出包含x的叶节点
        # （递归）回退找寻其他可能节点
        dx = x[kdtree.fi] - kdtree.fv
        if dx < 0:
            self.nearest(kdtree.left, x, k, best_k)
            if x[kdtree.fi] + best_k[-1][0] > kdtree.fv:
                self.nearest(kdtree.right, x, k, best_k)
        elif dx > 0:
            self.nearest(kdtree.right, x, k, best_k)
            if x[kdtree.fi] - best_k[-1][0] < kdtree.fv :
                self.nearest(kdtree.left, x, k, best_k)
        else:
            self.nearest(kdtree.left, x, k, best_k)
            self.nearest(kdtree.right, x, k, best_k)
        return best_k
    
    def k_nearest(self, x):
        best_k = self.nearest(self.tree, x, self.k, None)
        # best_k_point =  [ best[1:] for best in best_k]  # 列表推导式
        return best_k
    
    def _predict(self,x):
        ''' 
        :param x 是equery的一个输入
        '''
        best_k = self.k_nearest(x)
        labels = [best[2] for best in best_k]
        
        counter={}
        for i in labels:
            counter.setdefault(i,0)
            counter[i] += 1         # i的值 +1
        sort=sorted(counter.items(),key=lambda x:x[1])
        return sort[-1][0]
    
    def predict(self,equery):
        return np.array([self._predict(x) for x in tqdm(equery)])
    
