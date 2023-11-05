import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from solve import solve

X, y = load_breast_cancer(return_X_y=True)
y[y == 0] = -1  # Label should in {-1, +1}.

train_X, test_X, train_y, test_y = train_test_split(
    X,
    y,
    train_size=0.7,
    random_state=42,
)
scaler = MinMaxScaler().fit(train_X)
train_X, test_X = scaler.transform(train_X), scaler.transform(test_X)

_, d = train_X.shape
Nw = 20000
W = np.random.randn(d, Nw)
b = np.random.rand(Nw) * 2 * np.pi
Phi = np.cos(train_X @ W + b)
Ks = np.square(Phi.T @ train_y)

# Construct vanilla random feature
train_unscale = np.cos(train_X @ W + b)
test_unscale = np.cos(test_X @ W + b)
train_rf1 = train_unscale / np.sqrt(Nw)
test_rf1 = test_unscale / np.sqrt(Nw)

# Train the model on vanilla random feature
model1 = LogisticRegression().fit(train_rf1, train_y)
acc1 = model1.score(test_rf1, test_y)

# Get the distribution q
rho = Nw * 0.005
q = solve(-Ks, np.ones(Nw) / Nw, rho / Nw)

# Delete the feature with q_i=0.
sparse = (q != 0.)
train_rf2 = train_unscale[:, sparse] * np.sqrt(q[sparse])
test_rf2 = test_unscale[:, sparse] * np.sqrt(q[sparse])

# train the model on optmized random feature
model2 = LogisticRegression().fit(train_rf2, train_y)
acc2 = model2.score(test_rf2, test_y)

print("Before optimization: {:.4f}, after optimization: {:.4f}".format(
    acc1, acc2))
