import numpy as np
import matplotlib.pyplot as plt

class Data(object):

    def __init__(self, path):
        self.dataset = self._build_dataset(path)

    def _build_dataset(self, path):
        structure = [('income', np.float),
                     ('happiness', np.float)]

        with open(path, encoding="utf8") as data_csv:

            data_gen = ((float(line.split(',')[1]), float(line.split(',')[2])) # add here + 10 in second value
                        for i, line in enumerate(data_csv) if i != 0)
            embeddings = np.fromiter(data_gen, structure)

        return embeddings

    def split(self, percentage): # 0.8
        X = self.dataset['income']
        y = self.dataset['happiness']

        # X.shape[0] -> 10 (filas)

        permuted_idxs = np.random.permutation(X.shape[0])
        # 2,1,3,4,6,7,8,5,9,0

        train_idxs = permuted_idxs[0:int(percentage * X.shape[0])]
                     #permuted_idxs[0:8]
                     #[2,1,3,4,5,6,7,8,5]

        test_idxs = permuted_idxs[int(percentage * X.shape[0]): X.shape[0]]
                    #[9,0]

        X_train = X[train_idxs]
        X_test = X[test_idxs]

        y_train = y[train_idxs]
        y_test = y[test_idxs]

        return X_train, X_test, y_train, y_test


class BaseModel(object):

    def __init__(self):
        self.model = None

    def fit(self, X, Y):
        return NotImplemented

    def predict(self, X):
        return NotImplemented


class ConstantModel(BaseModel):

    def fit(self, X, Y):
        W = Y.mean()
        self.model = W

    def predict(self, X):
        return np.ones(len(X)) * self.model


class LinearRegression(BaseModel):

    def fit(self, X, y):
        if len(X.shape) == 1:
            W = X.T.dot(y) / X.T.dot(X)
        else:
            W = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)
        self.model = W

    def predict(self, X):
        return self.model * X


class LinearRegressionWithB(BaseModel):

    def fit(self, X, y):
        X_expanded = np.vstack((X, np.ones(len(X)))).T
        W = np.linalg.inv(X_expanded.T.dot(X_expanded)).dot(X_expanded.T).dot(y)
        self.model = W

    def predict(self, X):
        X_expanded = np.vstack((X, np.ones(len(X)))).T
        return X_expanded.dot(self.model)


class Metric(object):
    def __call__(self, target, prediction):
        return NotImplemented


class MSE(Metric):
    def __call__(self, target, prediction):
        n = target.size
        return np.sum((target - prediction) ** 2) / n


if __name__ == '__main__':
    dataset = Data('./clase_3/income.data.csv')

    X_train, X_test, y_train, y_test = dataset.split(0.8)

    linear_regression = LinearRegression()
    linear_regression.fit(X_train, y_train)
    lr_y_hat = linear_regression.predict(X_test)

    linear_regression_b = LinearRegressionWithB()
    linear_regression_b.fit(X_train, y_train)
    lrb_y_hat = linear_regression_b.predict(X_test)

    constant_model = ConstantModel()
    constant_model.fit(X_train, y_train)
    ct_y_hat = constant_model.predict(X_test)

    mse = MSE()
    lr_mse = mse(y_test, lr_y_hat)
    lrb_mse = mse(y_test, lrb_y_hat)
    ct_mse = mse(y_test, ct_y_hat)

    x_plot = np.linspace(0, 10, 10)
    lr_y_plot = linear_regression.model * x_plot
    lrb_y_plot = linear_regression_b.model[0] * x_plot + linear_regression_b.model[1]

    plt.scatter(X_train, y_train, color='b', label='dataset')
    plt.plot(x_plot, lr_y_plot, color='m', label=f'LinearRegresion(MSE={lr_mse:.3f})')
    plt.plot(x_plot, lrb_y_plot, color='r', label=f'LinearRegresionWithB(MSE={lrb_mse:.3f})')
    plt.plot(X_test, ct_y_hat, color='g', label=f'ConstantModel(MSE={ct_mse:.3f})')
    plt.legend()
    plt.show()