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

    # initialize random weights
    W = np.random.randn(m).reshape(m, 1)

    for i in range(amt_epochs):
        prediction = np.matmul(X_train, W)  # nx1
        error = y_train - prediction  # nx1

        grad_sum = np.sum(error * X_train, axis=0)
        grad_mul = -2/n * grad_sum  # 1xm
        gradient = np.transpose(grad_mul).reshape(-1, 1)  # mx1

        W = W - (lr * gradient)

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

    # initialize random weights
    W = np.random.randn(m).reshape(m, 1)

    for i in range(amt_epochs):
        idx = np.random.permutation(X_train.shape[0])
        X_train = X_train[idx]
        y_train = y_train[idx]

        for j in range(n):
            prediction = np.matmul(X_train[j].reshape(1, -1), W)  # 1x1
            error = y_train[j] - prediction  # 1x1

            grad_sum = error * X_train[j]
            grad_mul = -2/n * grad_sum  # 2x1
            gradient = np.transpose(grad_mul).reshape(-1, 1)  # 2x1

            W = W - (lr * gradient)

    return W


def mini_batch_gradient_descent(X_train, y_train, lr=0.01, amt_epochs=100):
    """
    shapes:
        X_t = nxm
        y_t = nx1
        W = mx1
    """
    b = 16
    n = X_train.shape[0]
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

            prediction = np.matmul(batch_X, W)  # nx1
            error = batch_y - prediction  # nx1

            grad_sum = np.sum(error * batch_X, axis=0)
            grad_mul = -2/n * grad_sum  # 1xm
            gradient = np.transpose(grad_mul).reshape(-1, 1)  # mx1

            W = W - (lr * gradient)

    return W


def sin_fitting_example():
    # y = sin(x)
    amt_points = 36
    x = np.linspace(0, 360, num=amt_points)
    y = np.sin(x * np.pi / 180.)
    noise = np.random.normal(0, .1, y.shape)
    noisy_y = y + noise

    X_train = x
    y_train = noisy_y

    regression = LinearRegression()

    # linear
    X_linear = np.vstack((X_train, np.ones(len(X_train)))).T
    regression.fit(X_linear, y_train.reshape(-1, 1))
    W_linear = regression.model
    y_linear = W_linear[0]*x + W_linear[1]

    # quadratic
    X_quadratic = np.vstack((np.power(X_train, 2), X_train, np.ones(len(X_train)))).T
    regression.fit(X_quadratic, y_train.reshape(-1, 1))
    W_quadratic = regression.model
    y_quadratic = W_quadratic[0] * np.power(x, 2) + W_quadratic[1] * x + W_quadratic[2]

    # cubic
    X_cubic = np.vstack((np.power(X_train, 3), np.power(X_train, 2), X_train, np.ones(len(X_train)))).T
    regression.fit(X_cubic, y_train.reshape(-1, 1))
    W_cubic = regression.model
    y_cubic = W_cubic[0] * np.power(x, 3) + W_cubic[1] * np.power(x, 2) + W_cubic[2] * x + W_cubic[3]

    # X10
    X_10 = np.vstack((np.power(X_train, 10), np.power(X_train, 9), np.power(X_train, 8),
                      np.power(X_train, 7), np.power(X_train, 6), np.power(X_train, 5),
                      np.power(X_train, 4), np.power(X_train, 3), np.power(X_train, 2),
                      X_train, np.ones(len(X_train)))).T
    regression.fit(X_10, y_train.reshape(-1, 1))
    W_10 = regression.model
    y_10 = W_10[0] * np.power(x, 10) + W_10[1] * np.power(x, 9) + W_10[2] * np.power(x, 8) + \
           W_10[3] * np.power(x, 7) + W_10[4] * np.power(x, 6) + W_10[5] * np.power(x, 5) + \
           W_10[6] * np.power(x, 4) + W_10[7] * np.power(x, 3) + W_10[8] * np.power(x, 2) + \
           W_10[9] * x + W_10[10]

    # PLOTS
    plt.figure()
    plt.subplot(1, 1, 1)
    plt.gca().set_title('Sin(x) - Fitting curves')

    # original
    plt.plot(x, noisy_y, 'o')

    # linear
    plt.plot(x, y_linear, '-')

    # quadratic
    plt.plot(x, y_quadratic, '-')

    # cubic
    plt.plot(x, y_cubic, '-')

    # 10 power
    plt.plot(x, y_10, '-')

    plt.legend(['noisy signal', 'linear', 'quadratic', 'cubic', '10th power'])
    plt.show()


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

    # gradient descent
    print('\n\n\nGRADIENT DESCENT VS LINEAR REGRESSION')
    lr_1 = 0.001
    amt_epochs_1 = 1000
    start = time.time()
    W_manual = gradient_descent(X_train.reshape(-1, 1), y_train.reshape(-1, 1), lr=lr_1, amt_epochs=amt_epochs_1)
    time_1 = time.time() - start
    W_real = linear_regression.model
    print('W_manual:  {}\nW_real:    {}\nManual time [s]: {}'.format(W_manual.reshape(-1), W_real, time_1))

    # gradient descent
    print('\n\n\nGRADIENT DESCENT VS LINEAR REGRESSION WITH B')
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
    print('\n\n\nSTOCHASTIC GRADIENT DESCENT VS LINEAR REGRESSION WITH B')
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
    print('\n\n\nMINI BATCH GRADIENT DESCENT VS LINEAR REGRESSION WITH B')
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
    plt.figure()
    x_plot = np.linspace(1, 4, 4)
    legend = ['GD', 'GD(B)', 'S-GD(B)', 'MB-GD(B)']

    plt.subplot(1, 3, 1)
    plt.gca().set_title('Learning Rate')
    y_plot = [lr_1, lr_2, lr_3, lr_4]
    plt.plot(x_plot[0], y_plot[0], 'o', x_plot[1], y_plot[1], 'o', x_plot[2], y_plot[2], 'o',
             x_plot[3], y_plot[3], 'o')
    plt.legend(legend)
    for x, y in zip(x_plot, y_plot):
        plt.text(x, y, str(y))

    plt.subplot(1, 3, 2)
    plt.gca().set_title('Epochs')
    y_plot = [amt_epochs_1, amt_epochs_2, amt_epochs_3, amt_epochs_4]
    plt.plot(x_plot[0], y_plot[0], 'o', x_plot[1], y_plot[1], 'o', x_plot[2], y_plot[2], 'o',
             x_plot[3], y_plot[3], 'o')
    plt.legend(legend)
    for x, y in zip(x_plot, y_plot):
        plt.text(x, y, str(y))

    plt.subplot(1, 3, 3)
    plt.gca().set_title('Time')
    y_plot = [time_1, time_2, time_3, time_4]
    plt.plot(x_plot[0], y_plot[0], 'o', x_plot[1], y_plot[1], 'o', x_plot[2], y_plot[2], 'o',
             x_plot[3], y_plot[3], 'o')
    plt.legend(legend)
    for x, y in zip(x_plot, y_plot):
        plt.text(x, y, str(y))

    plt.show()

    # sin fitting example
    sin_fitting_example()
