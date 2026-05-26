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
    lr = 0.1

    for iteration in range(100):
        f = x ** 2
        f.backward()
        with torch.no_grad():
            x -= lr * x.grad

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
    return


if __name__ == "__main__":
    app.run()
