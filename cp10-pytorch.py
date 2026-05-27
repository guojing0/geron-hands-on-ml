import marimo

__generated_with = "0.23.8"
app = marimo.App(width="medium")


@app.cell
def _():
    import torch

    return (torch,)


@app.cell
def _(torch):
    X = torch.tensor([[1.0, 4.0, 7.0], [2.0, 3.0, 6.0]])
    X
    return (X,)


@app.cell
def _(X):
    print(X.shape)
    print(X.dtype)
    return


@app.cell
def _(X):
    X[:, 1]
    return


@app.cell
def _(X):
    X.max(dim=0)
    return


@app.cell
def _(X):
    X @ X.T
    return


@app.cell
def _(X):
    X[:, 1] = -42
    X.relu_()
    return


@app.cell
def _(torch):
    import timeit

    device = "mps"

    M = torch.rand((5000, 5000))
    print(timeit.timeit("M @ M.T", globals=globals(), number=10))

    M = torch.rand((5000, 5000), device=device)
    print(timeit.timeit("M @ M.T", globals=globals(), number=10))

    return


@app.cell
def _(torch):
    x = torch.tensor(5.0, requires_grad=True)
    _lr = 0.1

    for iteration in range(100):
        f = x ** 2
        f.backward()
        with torch.no_grad():
            x -= _lr * x.grad

        x.grad.zero_()

        print(x)
    return


@app.cell
def _(torch):
    t = torch.tensor(2.0, requires_grad=True)
    z = t.exp()
    z = z + 1
    z.backward()
    return


@app.cell
def _():
    # Linear regression with tensors and autograd

    from sklearn.datasets import fetch_california_housing
    from sklearn.model_selection import train_test_split

    housing = fetch_california_housing()
    return housing, train_test_split


@app.cell
def _(housing, torch, train_test_split):
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        housing.data,
        housing.target,
        random_state=42
    )

    X_train, X_valid, y_train, y_valid = train_test_split(
        X_train_full,
        y_train_full,
        random_state=42
    )

    # Convert to tensors
    X_train = torch.FloatTensor(X_train)
    X_valid = torch.FloatTensor(X_valid)
    X_test = torch.FloatTensor(X_test)

    means = X_train.mean(dim=0, keepdim=True)
    stds = X_train.std(dim=0, keepdim=True)

    # Normalization
    X_train = (X_train - means) / stds
    X_valid = (X_valid - means) / stds
    X_test = (X_test - means) / stds

    # Convert targets to tensors
    y_train = torch.FloatTensor(y_train).reshape(-1, 1)
    y_valid = torch.FloatTensor(y_valid).reshape(-1, 1)
    y_test = torch.FloatTensor(y_test).reshape(-1, 1)
    return X_test, X_train, y_train


@app.cell
def _(X_train, torch, y_train):
    # Prepare parameters for linear regression model

    torch.manual_seed(42)
    n_features = X_train.shape[1]
    w = torch.randn((n_features, 1), requires_grad=True)
    b = torch.tensor(0., requires_grad=True)

    lr = 0.05
    n_epoches = 500

    for epoch in range(n_epoches):
        y_pred = X_train @ w + b
        loss = ((y_pred - y_train) ** 2).mean()
        loss.backward()

        with torch.no_grad():
            b -= lr * b.grad
            w -= lr * w.grad
            b.grad.zero_()
            w.grad.zero_()
        print(f"Epoch {epoch + 1} / {n_epoches}, Loss: {loss.item()}")
    return b, lr, n_epoches, n_features, w


@app.cell
def _(X_test, b, torch, w):
    X_new = X_test[:3]
    with torch.no_grad():
        _y_pred = X_new @ w + b
    print(_y_pred)
    return


@app.cell
def _(lr, n_features, torch):
    # Linear regression with high-level API

    import torch.nn as nn

    torch.manual_seed(42)

    model = nn.Linear(in_features=n_features, out_features=1)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    mse = nn.MSELoss()
    return model, mse, optimizer


@app.function
def train_bgd(model, optimizer, criterion, X_train, y_train, n_epoches):
    for epoch in range(n_epoches):
        y_pred = model(X_train)
        loss = criterion(y_pred, y_train)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        print(f"Epoch {epoch + 1}/{n_epoches}, Loss: {loss.item()}")


@app.cell
def _(X_train, model, mse, n_epoches, optimizer, y_train):
    train_bgd(model, optimizer, mse, X_train, y_train, n_epoches)
    return


@app.cell
def _(X_test, model, torch):
    _X_new = X_test[:3]

    with torch.no_grad():
        _y_pred = model(_X_new)

    print(_y_pred)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
