import numpy as np
from tqdm import tqdm

class MyAdaboost:
    def __init__(self, n_estimators):
        self.n_estimators = n_estimators
        self.weights = None
        self.clfs = [lambda x:0 for i in range(n_estimators)]
        self.alphas = [0 for i in range(n_estimators)]

    def _G(self, fi, fv, direction):
        assert direction in ['positive', 'negative']
        def _g(X):
            if direction == 'positive':
                predict = (X[:,fi] <= fv) * -1
            else:
                predict = (X[:,fi] > fv) * -1
            predict[predict == 0] = 1
            return predict
        return _g

    def _best_split(self, X, y, w):
        best_err = X.shape[0]
        best_fi = None
        best_fv = None
        best_direction = None
        for fi in range(X.shape[1]):
            series = X[:,fi]
            for fv in np.sort(series):
                predict = np.zeros_like(series, dtype=np.int32)
                # direction = positive
                predict[series <= fv] = -1
                predict[series > fv] = 1
                err = np.sum( (predict != y)*w )
                if err < best_err:
                    best_err = err
                    best_fi = fi
                    best_fv = fv
                    best_direction = 'positive'
                predict = predict * -1
                err = np.sum( (predict != y)*w )
                if err < best_err:
                    best_err = err
                    best_fi = fi
                    best_fv = fv
                    best_direction = 'negative'
        return best_err, best_fi, best_fv, best_direction
    
    def fit(self, X_train, y_train):
        self.weights = np.ones_like(y_train) / len(y_train)
        for i in tqdm(range(self.n_estimators)):
            err, fi, fv, direction = self._best_split(X_train, y_train, self.weights)
            if err == 0: break
            # 计算G(x)
            g = self._G(fi, fv, direction)
            self.clfs[i] = g
            # 计算 G(x)的系数alpha
            alpha = 0.5 * np.log((1-err)/err)
            self.alphas[i] = alpha
            # 更新weights
            self.weights = self.weights * np.exp( -1 * alpha * y_train * g(X_train) )
            self.weights = self.weights / np.sum(self.weights)
    
    def predict(self, X_test):
        y_p = np.array([self.alphas[i] * self.clfs[i](X_test) for i in range(self.n_estimators)])
        y_p = np.sum(y_p, axis=0)
        y_predict = np.zeros_like(y_p, dtype=np.int32)
        y_predict[y_p>=0] = 1
        y_predict[y_p<0] = -1
        return y_predict
    
    def score(self, X_test, y_test):
        y_predict = self.predict(X_test)
        return np.sum(y_predict == y_test) / len(y_test)
    
if __name__ == '__main__':
    from sklearn.datasets import load_breast_cancer
    from sklearn.model_selection import train_test_split

    breast_cancer = load_breast_cancer()
    X = breast_cancer.data
    y = breast_cancer.target
    y[y==0] = -1
    # 划分数据
    X_train, X_test, y_train, y_test = train_test_split(X,y)

    clf = MyAdaboost(100)
    clf.fit(X_train,y_train)
    print(clf.score(X_test,y_test))