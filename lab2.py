import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDRegressor, Ridge, LinearRegression, RANSACRegressor

X_train, X_test, Y_train, Y_test, graphX, graphY, graphXBB, graphYBB = [], [], [], [], [], [], [], []

def rmse(y_pred, y_actual):
    return (sum([(y_pred_i - y_actual_i)**2 for y_pred_i, y_actual_i in zip(y_pred, y_actual)])/len(y_actual))**(1/2)

def nrmse(y_pred, y_actual):
    return rmse(y_pred, y_actual)/(max(y_actual) - min(y_actual))


with open("1_train.txt", "r") as file:
    for line in file:
        newline = line.rstrip('\n')
        X_train.append([int(i) for i in newline.split(' ')])
        Y_train += [X_train[len(X_train)-1].pop()]

with open("1_test.txt", "r") as file:
    for line in file:
        newline = line.rstrip('\n')
        X_test.append([int(i) for i in newline.split(' ')])
        Y_test += [X_test[len(X_test)-1].pop()]

X_train = preprocessing.normalize(X_train)
X_test = preprocessing.normalize(X_test)
Y_train, Y_test = np.array(Y_train), np.array(Y_test)
np.hstack((X_train, np.ones((X_train.shape[0], 1))))
np.hstack((X_test, np.ones((X_test.shape[0], 1))))
# modelLS = Ridge(alpha=0.5, solver='svd')
# modelLS.fit(X_train, Y_train)
# y = modelLS.predict(X_train)
# error1 = nrmse(y, Y_train)
# print(error1)
# y_pred = modelLS.predict(X_test)
# error2 = nrmse(y_pred, Y_test)
# print(error2)

# for i in range(100, 300, 50):
#     modelGD = SGDRegressor(shuffle=True, max_iter=i, penalty="elasticnet", alpha=0.01, learning_rate="invscaling", eta0=0.001, l1_ratio=0.6, power_t=0.3)
#     modelGD.fit(X_train, Y_train)
#     y_pred = modelGD.predict(X_test)
#     graphX.append(i)
#     graphY.append(nrmse(y_pred, Y_test))
#     print(nrmse(y_pred, Y_test))
#
# plt.plot(graphX, graphY, label="nrmse error for iter number")
# plt.xlabel("iterations")
# plt.ylabel("NRMSE")
# plt.legend()
# plt.show()

for i in range(100, 5000, 100):
    modelBB = RANSACRegressor(max_trials=i, max_skips=100, stop_score=0.95)
    modelBB.fit(X_train, Y_train)
    y_pred = modelBB.predict(X_test)
    graphXBB.append(i)
    graphYBB.append(nrmse(y_pred, Y_test))
    print(nrmse(y_pred, Y_test))

plt.plot(graphXBB, graphYBB, label="nrmse error for sample number")
plt.xlabel("samples")
plt.ylabel("NRMSE")
plt.legend()
plt.show()