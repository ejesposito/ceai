import time

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
    def __init__(self):
        Metric.__init__(self)

    def __call__(self, target, prediction):
        n = target.size
        return np.sum((target - prediction) ** 2) / n


def k_folds(X_train, y_train, k=5):
    l_regression = LinearRegression()
    error = MSE()

    chunk_size = int(len(X_train) / k)
    mse_list = []
    for i in range(0, len(X_train), chunk_size):
        end = i + chunk_size if i + chunk_size <= len(X_train) else len(X_train)
        new_X_valid = X_train[i: end]
        new_y_valid = y_train[i: end]
        new_X_train = np.concatenate([X_train[: i], X_train[end:]])
        new_y_train = np.concatenate([y_train[: i], y_train[end:]])

        l_regression.fit(new_X_train, new_y_train)
        prediction = l_regression.predict(new_X_valid)
        mse_list.append(error(new_y_valid, prediction))

    mean_MSE = np.mean(mse_list)

    return mean_MSE


def gradient_descent(X_train, y_train, lr=0.01, amt_epochs=100):
    """
    shapes:
        X_t = nxm
        y_t = nx1
        W = mx1
    """
    n = X_train.shape[0]
    m = X_train.shape[1]
    print('X.shape: {} x {} \n'.format(n, m))

    # initialize random weights
    W = np.random.randn(m).reshape(m, 1)
    # W = np.random.uniform(0, 1, (m, 1))
    # W = np.array([0.70, 0.2]).reshape(m, 1)
    print('W_inicial: {}'.format(W.reshape(-1)))

    for i in range(amt_epochs):
        prediction = np.matmul(X_train, W)  # nx1
        # print('pred', prediction.shape)
        error = y_train - prediction  # nx1
        # print('error', error.shape)

        grad_sum = np.sum(error * X_train, axis=0)
        grad_mul = -2/n * grad_sum  # 1xm
        gradient = np.transpose(grad_mul).reshape(-1, 1)  # mx1
        # print('grad', gradient.shape)

        W = W - (lr * gradient)
        # print('w', W.shape)
        # print('W_intermedio: ', W)

    return W


def stochastic_gradient_descent(X_train, y_train, lr=0.01, amt_epochs=100):
    """
    shapes:
        X_t = nxm
        y_t = nx1
        W = mx1
    """
    n = X_train.shape[0]
    m = X_train.shape[1]
    print('X.shape: {} x {} \n'.format(n, m))

    # initialize random weights
    W = np.random.randn(m).reshape(m, 1)
    # W = np.random.uniform(0, 1, (m, 1))
    # W = np.array([0.70, 0.2]).reshape(m, 1)
    print('W_inicial: {}'.format(W.reshape(-1)))

    for i in range(amt_epochs):
        idx = np.random.permutation(X_train.shape[0])
        X_train = X_train[idx]
        y_train = y_train[idx]

        for j in range(n):
            prediction = np.matmul(X_train[j].reshape(1, -1), W)  # 1x1
            # print('pred', prediction.shape)
            error = y_train[j] - prediction  # 1x1
            # print('error', error.shape)

            grad_sum = error * X_train[j]
            grad_mul = -2/n * grad_sum  # 2x1
            gradient = np.transpose(grad_mul).reshape(-1, 1)  # 2x1
            # print('grad', gradient.shape)

            W = W - (lr * gradient)
            # print('w', W.shape)
            # print('W_intermedio: ', W)

    return W


def mini_batch_gradient_descent(X_train, y_train, lr=0.01, amt_epochs=100):
    """
    shapes:
        X_t = nxm
        y_t = nx1
        W = mx1
    """
    b = 15
    n = X_train.shape[0]
    m = X_train.shape[1]
    print('X.shape: {} x {} \n'.format(n, m))

    # initialize random weights
    W = np.random.randn(m).reshape(m, 1)
    # W = np.random.uniform(0, 1, (m, 1))
    # W = np.array([0.70, 0.2]).reshape(m, 1)
    print('W_inicial: {}'.format(W.reshape(-1)))

    for i in range(amt_epochs):
        idx = np.random.permutation(X_train.shape[0])
        X_train = X_train[idx]
        y_train = y_train[idx]

        batch_size = int(len(X_train) / b)
        for i in range(0, len(X_train), batch_size):
            end = i + batch_size if i + batch_size <= len(X_train) else len(X_train)
            batch_X = X_train[i: end]
            batch_y = y_train[i: end]

            prediction = np.matmul(batch_X, W)  # nx1
            # print('pred', prediction.shape)
            error = batch_y - prediction  # nx1
            # print('error', error.shape)

            grad_sum = np.sum(error * batch_X, axis=0)
            grad_mul = -2/n * grad_sum  # 1xm
            gradient = np.transpose(grad_mul).reshape(-1, 1)  # mx1
            # print('grad', gradient.shape)

            W = W - (lr * gradient)
            # print('w', W.shape)
            # print('W_intermedio: ', W)

    return W


if __name__ == '__main__':
    dataset = Data('./clase_4/income.data.csv')

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

    """
    plt.scatter(X_train, y_train, color='b', label='dataset')
    plt.plot(x_plot, lr_y_plot, color='m', label=f'LinearRegresion(MSE={lr_mse:.3f})')
    plt.plot(x_plot, lrb_y_plot, color='r', label=f'LinearRegresionWithB(MSE={lrb_mse:.3f})')
    plt.plot(X_test, ct_y_hat, color='g', label=f'ConstantModel(MSE={ct_mse:.3f})')
    plt.legend()
    plt.show()
    """

    # gradient descent
    print('\nGRADIENT DESCENT VS LINEAR REGRESSION')
    lr_1 = 0.001
    amt_epochs_1 = 1000
    start = time.time()
    W_manual = gradient_descent(X_train.reshape(-1, 1), y_train.reshape(-1, 1), lr=lr_1, amt_epochs=amt_epochs_1)
    time_1 = time.time() - start
    W_real = linear_regression.model
    print('W_manual:  {}\nW_real:    {}\nManual time [s]: {}'.format(W_manual.reshape(-1), W_real, time_1))

    # gradient descent
    print('\nGRADIENT DESCENT VS LINEAR REGRESSION WITH B')
    X_expanded = np.vstack((X_train, np.ones(len(X_train)))).T
    lr_2 = 0.001
    amt_epochs_2 = 100000
    start = time.time()
    W_manual = gradient_descent(X_expanded, y_train.reshape(-1, 1), lr=lr_2, amt_epochs=amt_epochs_2)
    time_2 = time.time() - start
    W_real = linear_regression_b.model
    print('W_manual:  {}\nW_real:    {}\nManual time [s]: {}'.
          format(W_manual.reshape(-1), W_real, time_2))

    # gradient descent
    print('\nSTOCHASTIC GRADIENT DESCENT VS LINEAR REGRESSION WITH B')
    X_expanded = np.vstack((X_train, np.ones(len(X_train)))).T
    lr_3 = 0.05
    amt_epochs_3 = 1000
    start = time.time()
    W_manual = stochastic_gradient_descent(X_expanded, y_train.reshape(-1, 1), lr=lr_3, amt_epochs=amt_epochs_3)
    time_3 = time.time() - start
    W_real = linear_regression_b.model
    print('W_manual:  {}\nW_real:    {}\nManual time [s]: {}'.
          format(W_manual.reshape(-1), W_real, time_3))

    # gradient descent
    print('\nMINI BATCH GRADIENT DESCENT VS LINEAR REGRESSION WITH B')
    X_expanded = np.vstack((X_train, np.ones(len(X_train)))).T
    lr_4 = 0.05
    amt_epochs_4 = 10000
    start = time.time()
    W_manual = mini_batch_gradient_descent(X_expanded, y_train.reshape(-1, 1), lr=lr_4, amt_epochs=amt_epochs_4)
    time_4 = time.time() - start
    W_real = linear_regression_b.model
    print('W_manual:  {}\nW_real:    {}\nManual time [s]: {}'.
          format(W_manual.reshape(-1), W_real, time_4))

    # PLOTS
    fig = plt.figure()
    x_plot = np.linspace(1, 4, 4)

    plt.subplot(1, 3, 1)
    plt.gca().set_title('Learning Rate')
    y_plot = [lr_1, lr_2, lr_3, lr_4]
    plt.plot(x_plot[0], y_plot[0], 'o', x_plot[1], y_plot[1], 'o', x_plot[2], y_plot[2], 'o',
             x_plot[3], y_plot[3], 'o')
    plt.legend(['GD', 'GD(B)', 'S-GD(B)', 'MB-GD(B)'])
    for x, y in zip(x_plot, y_plot):
        plt.text(x, y, str(y))

    plt.subplot(1, 3, 2)
    plt.gca().set_title('Epochs')
    y_plot = [amt_epochs_1, amt_epochs_2, amt_epochs_3, amt_epochs_4]
    plt.plot(x_plot[0], y_plot[0], 'o', x_plot[1], y_plot[1], 'o', x_plot[2], y_plot[2], 'o',
             x_plot[3], y_plot[3], 'o')
    plt.legend(['GD', 'GD(B)', 'S-GD(B)', 'MB-GD(B)'])
    for x, y in zip(x_plot, y_plot):
        plt.text(x, y, str(y))

    plt.subplot(1, 3, 3)
    plt.gca().set_title('Time')
    y_plot = [time_1, time_2, time_3, time_4]
    plt.plot(x_plot[0], y_plot[0], 'o', x_plot[1], y_plot[1], 'o', x_plot[2], y_plot[2], 'o',
             x_plot[3], y_plot[3], 'o')
    plt.legend(['GD', 'GD(B)', 'S-GD(B)', 'MB-GD(B)'])
    for x, y in zip(x_plot, y_plot):
        plt.text(x, y, str(y))

    plt.show()


