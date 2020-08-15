import time

import numpy as np
import matplotlib.pyplot as plt


class Data(object):

    def __init__(self, path):
        self.dataset = self._build_dataset(path)

    def _build_dataset(self, path):
        structure = [('exam_1', np.float),
                     ('exam_2', np.float),
                     ('admission', np.int)]

        with open(path, encoding="utf8") as data_csv:
            data_gen = ((float(line.split(',')[0]), float(line.split(',')[1]),
                         np.int(line.split(',')[2]))
                        for i, line in enumerate(data_csv) if i != 0)
            embeddings = np.fromiter(data_gen, structure)

        return embeddings

    def split(self, percentage):
        X = self.dataset[['exam_1', 'exam_2']]
        y = self.dataset['admission']

        permuted_idxs = np.random.permutation(len(X))

        train_idxs = permuted_idxs[0:int(percentage * X.shape[0])]

        test_idxs = permuted_idxs[int(percentage * X.shape[0]): X.shape[0]]

        X_train = X[train_idxs]
        X_test = X[test_idxs]

        y_train = y[train_idxs]
        y_test = y[test_idxs]

        return X_train, X_test, y_train, y_test


class Metric(object):
    def __call__(self, target, prediction):
        return NotImplemented


class MSE(Metric):
    def __init__(self):
        Metric.__init__(self)

    def __call__(self, target, prediction):
        n = target.size
        return np.sum((target - prediction) ** 2) / n


def mini_batch_logistic_regression(X_train, y_train, lr=0.01, amt_epochs=100):
    """
    shapes:
        X_t = nxm
        y_t = nx1
        W = mx1
    """
    b = 16
    m = X_train.shape[1]

    # initialize random weights
    W = np.random.randn(m).reshape(m, 1)

    for i in range(amt_epochs):
        idx = np.random.permutation(X_train.shape[0])
        X_train = X_train[idx]
        y_train = y_train[idx]

        batch_size = int(len(X_train) / b)
        for i in range(0, len(X_train), batch_size):
            end = i + batch_size if i + batch_size <= len(X_train) else len(X_train)
            batch_X = X_train[i: end]
            batch_y = y_train[i: end]

            exponent = np.sum(np.transpose(W) * batch_X, axis=1)
            prediction = 1/(1 + np.exp(-exponent))
            error = prediction.reshape(-1, 1) - batch_y.reshape(-1, 1)

            grad_sum = np.sum(error * batch_X, axis=0)
            grad_mul = 1/b * grad_sum
            gradient = np.transpose(grad_mul).reshape(-1, 1)

            W = W - (lr * gradient)

    return W


if __name__ == '__main__':
    dataset = Data('./clase_6_dataset.csv')
    print('Dataset loaded')
    X_train, X_test, y_train, y_test = dataset.split(1)
    print('Dataset split')

    # gradient descent
    X_expanded = np.vstack((X_train['exam_1'], X_train['exam_2'], np.ones(len(X_train)))).T
    lr = 0.001
    amt_epochs = 50000
    print('Training')
    start = time.time()
    W = mini_batch_logistic_regression(X_expanded, y_train.reshape(-1, 1), lr=lr,
                                       amt_epochs=amt_epochs)
    time = time.time() - start
    print('W: {}\nTime [s]: {}'.format(W.reshape(-1), time))

    # PLOTS
    # filter out the applicants that got admitted
    admitted = X_train[y_train == 1]
    # filter out the applicants that didn't get admission
    not_admitted = X_train[y_train == 0]

    # logistic regression
    x_regression = np.linspace(30, 100, 70)
    y_regression = (-x_regression*W[0] - W[2])/W[1]

    # plots
    plt.scatter(admitted['exam_1'], admitted['exam_2'], s=10, label='Admitted')
    plt.scatter(not_admitted['exam_1'], not_admitted['exam_2'], s=10, label='Not Admitted')
    plt.plot(x_regression, y_regression, '-', color='green', label='Regression')
    plt.legend()
    plt.show()
