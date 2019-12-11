# 2019.12.12.v1
import numpy as np 
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

class myRegression():
    def __init__(self, regressor, num_class):
        self.regressor = regressor
        self.num_class = num_class
        self.class_list = np.zeros((num_class))

    def fit(self, X, Y):
        tmp = np.unique(Y)
        tmp = tmp.astype('int32')
        if tmp.shape[0] == self.num_class:
            self.class_list = np.ones((self.num_class))
        elif tmp.shape[0] == 1:
            self.class_list[tmp[0]] = 1
            return
        else:
            for i in range(tmp.shape[0]):
                Y[Y == tmp[i]] = i
                self.class_list[tmp[i]] = 1
        self.regressor.fit(X, Y)

    def predict_proba(self, X):
        if np.sum(self.class_list) == 1:
            return np.ones((X.shape[0], self.num_class)) * self.class_list
        try:
            tmp_pred = self.regressor.predict_proba(X)
        except:
            tmp_pred = self.regressor.predict(X).reshape(-1,1)
            tmp_pred = np.concatenate((tmp_pred, 1-tmp_pred), axis=1)
        pred = np.zeros((X.shape[0], self.num_class))
        idx = 0
        for i in range(self.num_class):
            if self.class_list[i] == 1:
                pred[:, i] = tmp_pred[:, idx]
                idx += 1
        return pred.reshape(X.shape[0], self.num_class)

    def score(self, X, Y):
        try:
            score = self.regressor.score(X, Y)
        except:
            pred = self.predict_proba(X)
            pred = np.argmax(pred, axis=1)
            score = accuracy_score(pred, Y)
        print("           <Debug Info>        accuracy on node: %s"%str(score))

if __name__ == "__main__":
    import cv2
    X = cv2.imread('../../data/test.jpg')
    X = cv2.resize(X, (40,40))
    X = X.reshape(-1,3)
    Y = np.random.randint(3, size=X.shape[0]) + 1
    Y = Y.reshape(-1,1)
    from sklearn.linear_model import LogisticRegression
    tmp = LogisticRegression(random_state=0, solver='liblinear', multi_class='ovr', n_jobs=20, max_iter=1000).fit(X, Y.reshape(-1,))
    reg = myRegression(tmp, 5)
    reg.fit(X, Y)
    pred = reg.predict_proba(X)
    print(" \n--> Regression result... \n", pred)