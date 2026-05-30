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
    return (device,)


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
    return X_test, X_train, X_valid, y_test, y_train, y_valid


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
    return b, epoch, lr, n_epoches, n_features, w, y_pred


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
    return model, mse, nn, optimizer


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
def _(n_features, nn, torch):
    torch.manual_seed(42)

    MLP_model = nn.Sequential(
        nn.Linear(n_features, 50),
        nn.ReLU(),
        nn.Linear(50, 40),
        nn.ReLU(),
        nn.Linear(40, 1)
    )
    return (MLP_model,)


@app.cell
def _(MLP_model, torch):
    MLP_lr = 0.01
    MLP_opt = torch.optim.SGD(MLP_model.parameters(), lr=MLP_lr)
    return MLP_lr, MLP_opt


@app.cell
def _(MLP_model, MLP_opt, X_train, mse, n_epoches, y_train):
    train_bgd(MLP_model, MLP_opt, mse, X_train, y_train, n_epoches)
    return


@app.cell
def _(X_train, device, y_train):
    from torch.utils.data import TensorDataset, DataLoader

    GPU_X_train = X_train.to(device)
    GPU_y_train = y_train.to(device)

    train_dataset = TensorDataset(GPU_X_train, GPU_y_train)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    return DataLoader, TensorDataset, train_loader


@app.cell
def _(MLP_lr, device, n_features, nn, torch):
    torch.manual_seed(42)

    GPU_MLP_model = nn.Sequential(
        nn.Linear(n_features, 50),
        nn.ReLU(),
        nn.Linear(50, 40),
        nn.ReLU(),
        nn.Linear(40, 1)
    )

    GPU_MLP_model = GPU_MLP_model.to(device)

    GPU_MLP_opt = torch.optim.SGD(GPU_MLP_model.parameters(), lr=MLP_lr)
    return GPU_MLP_model, GPU_MLP_opt


@app.function
# Mini-batch GD

def train(model, optimizer, criterion, train_loader, n_epoches):
    model.train()
    for epoch in range(n_epoches):
        total_loss = 0.
        for X_batch, y_batch in train_loader:
            # X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            y_pred = model(X_batch)

            loss = criterion(y_pred, y_batch)
            total_loss += loss.detach()

            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        # mean_loss = total_loss / len(train_loader)
        mean_loss = (total_loss / len(train_loader)).item()
        print(f"Epoch {epoch + 1}/{n_epoches}, Loss: {mean_loss:.4f}")


@app.cell
def _(GPU_MLP_model, GPU_MLP_opt, mse, train_loader):
    train(GPU_MLP_model, GPU_MLP_opt, mse, train_loader, 20)
    return


@app.cell
def _(device, torch):
    # Model eval

    def evaluate(model, data_loader, metric_fn, aggregate_fn=torch.mean):
        model.eval()
        metrics = []
        with torch.no_grad():
            for X_batch, y_batch in data_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                y_pred = model(X_batch)
                metric = metric_fn(y_pred, y_batch)
                metrics.append(metric)

        return aggregate_fn(torch.stack(metrics))

    return (evaluate,)


@app.cell
def _(
    DataLoader,
    GPU_MLP_model,
    TensorDataset,
    X_valid,
    evaluate,
    mse,
    y_valid,
):
    valid_dataset = TensorDataset(X_valid, y_valid)
    valid_loader = DataLoader(valid_dataset, batch_size=32)
    valid_mse = evaluate(GPU_MLP_model, valid_loader, mse)
    return valid_loader, valid_mse


@app.cell
def _(valid_mse):
    valid_mse
    return


@app.cell
def _(device, torch):
    import torchmetrics

    def evaluate_tm(model, data_loader, metric):
        model.eval()
        metric.reset()
        with torch.no_grad():
            for X_batch, y_batch in data_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                y_pred = model(X_batch)
                metric.update(y_pred, y_batch)
        return metric.compute()

    return evaluate_tm, torchmetrics


@app.cell
def _(device, torchmetrics):
    rmse = torchmetrics.MeanSquaredError(squared=False).to(device)
    return (rmse,)


@app.cell
def _(GPU_MLP_model, evaluate_tm, rmse, valid_loader):
    evaluate_tm(GPU_MLP_model, valid_loader, rmse)
    return


@app.cell
def _(GPU_MLP_model, X_train, device, y_pred):
    print(GPU_MLP_model(X_train[:5].to(device)))

    print(y_pred[:5])
    return


@app.cell
def _(nn, torch):
    # Non-sequential model (Wild & Deep)

    class WideAndDeep(nn.Module):

        def __init__(self, n_features):
            super().__init__()

            self.deep_stack = nn.Sequential(
                nn.Linear(n_features, 50),
                nn.ReLU(),
                nn.Linear(50, 40),
                nn.ReLU(),
            )
            self.output_layer = nn.Linear(40 + n_features, 1)

        def forward(self, X):
            deep_output = self.deep_stack(X)
            wide_and_deep = torch.concat([X, deep_output], dim=1)
            return self.output_layer(wide_and_deep)

    return (WideAndDeep,)


@app.cell
def _(WideAndDeep, device, n_features, torch):
    torch.manual_seed(42)
    WAD_model = WideAndDeep(n_features).to(device)
    WAD_lr = 0.002
    WAD_opt = torch.optim.SGD(WAD_model.parameters(), lr=WAD_lr)
    return WAD_model, WAD_opt


@app.cell
def _(WAD_model, WAD_opt, mse, train_loader):
    train(WAD_model, WAD_opt, mse, train_loader, 25)
    return


@app.cell
def _(nn, torch):
    class WideAndDeepV4(nn.Module):
        def __init__(self, n_features):
            super().__init__()
            self.deep_stack = nn.Sequential(
                nn.Linear(n_features - 2, 50),
                nn.ReLU(),
                nn.Linear(50, 40),
                nn.ReLU(),
                nn.Linear(40, 30),
                nn.ReLU(),
            )
            self.output_layer = nn.Linear(30 + 5, 1)
            self.aux_output_layer = nn.Linear(30, 1)

        def forward(self, X_wide, X_deep):
            deep_output = self.deep_stack(X_deep)
            wide_and_deep = torch.concat([X_wide, deep_output], dim=1)
            main_output = self.output_layer(wide_and_deep)
            aux_output = self.aux_output_layer(deep_output)
            return main_output, aux_output

    return (WideAndDeepV4,)


@app.cell
def _(torch):
    class WideAndDeepDataset(torch.utils.data.Dataset):
        def __init__(self, X_wide, X_deep, y):
            self.X_wide = X_wide
            self.X_deep = X_deep
            self.y = y

        def __len__(self):
            return len(self.y)

        def __getitem__(self, idx):
            input_dict = {"X_wide": self.X_wide[idx], "X_deep": self.X_deep[idx]}
            return input_dict, self.y[idx]

    return (WideAndDeepDataset,)


@app.cell
def _(
    DataLoader,
    WideAndDeepDataset,
    X_test,
    X_train,
    X_valid,
    y_test,
    y_train,
    y_valid,
):
    train_data_named = WideAndDeepDataset(
        X_wide=X_train[:, :5], X_deep=X_train[:, 2:], y=y_train)
    train_loader_named = DataLoader(train_data_named, batch_size=32, shuffle=True)

    valid_data_named = WideAndDeepDataset(
        X_wide=X_valid[:, :5], X_deep=X_valid[:, 2:], y=y_valid)
    valid_loader_named = DataLoader(valid_data_named, batch_size=32)

    test_data_named = WideAndDeepDataset(
        X_wide=X_test[:, :5], X_deep=X_test[:, 2:], y=y_test)
    test_loader_named = DataLoader(test_data_named, batch_size=32)
    return train_loader_named, valid_loader_named


@app.cell
def _(
    WideAndDeepV4,
    device,
    mse,
    n_features,
    rmse,
    torch,
    train_loader_named,
    valid_loader_named,
):
    def evaluate_multi_out(model, data_loader, metric):
        model.eval()
        metric.reset()
        with torch.no_grad():
            for inputs, y_batch in data_loader:
                inputs = {name: X.to(device) for name, X in inputs.items()}
                y_batch = y_batch.to(device)
                y_pred, _ = model(**inputs)
                metric.update(y_pred, y_batch)
        return metric.compute()

    def train_multi_out(model, optimizer, criterion, metric, train_loader,
                       valid_loader, n_epochs):
        history = {"train_losses": [], "train_metrics": [], "valid_metrics": []}
        for epoch in range(n_epochs):
            total_loss = 0.
            metric.reset()
            for inputs, y_batch in train_loader:
                model.train()
                inputs = {name: X.to(device) for name, X in inputs.items()}
                y_batch = y_batch.to(device)
                y_pred, y_pred_aux = model(**inputs)
                main_loss = criterion(y_pred, y_batch)
                aux_loss = criterion(y_pred_aux, y_batch)
                loss = 0.8 * main_loss + 0.2 * aux_loss
                total_loss += loss.item()
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                metric.update(y_pred, y_batch)
            mean_loss = total_loss / len(train_loader)
            history["train_losses"].append(mean_loss)
            history["train_metrics"].append(metric.compute().item())
            history["valid_metrics"].append(
                evaluate_multi_out(model, valid_loader, metric).item())
            print(f"Epoch {epoch + 1}/{n_epochs}, "
                  f"train loss: {history['train_losses'][-1]:.4f}, "
                  f"train metric: {history['train_metrics'][-1]:.4f}, "
                  f"valid metric: {history['valid_metrics'][-1]:.4f}")
        return history

    torch.manual_seed(42)
    WAD4_lr = 0.01
    WAD4_model = WideAndDeepV4(n_features).to(device)
    WAD4_opt = torch.optim.SGD(WAD4_model.parameters(), lr=WAD4_lr, momentum=0)
    history = train_multi_out(WAD4_model, WAD4_opt, mse, rmse, train_loader_named, valid_loader_named, 20)
    return


@app.cell
def _(torch):
    ### Image classifer for Fashion MNIST

    # Load the dataset

    import numpy
    import torchvision
    import torchvision.transforms.v2 as T

    toTensor = T.Compose([T.ToImage(), T.ToDtype(torch.float32, scale=True)])

    train_and_valid_data = torchvision.datasets.FashionMNIST(
        root="datasets", train=True, download=True, transform=toTensor
    )
    test_data = torchvision.datasets.FashionMNIST(
        root="datasets", train=False, download=True, transform=toTensor
    )

    torch.manual_seed(42)
    train_data, valid_data = torch.utils.data.random_split(
        train_and_valid_data, [55000, 5000]
    )
    return test_data, train_and_valid_data, train_data, valid_data


@app.cell
def _(DataLoader, test_data, train_data, valid_data):
    # Data loader

    fashion_train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    fashion_valid_loader = DataLoader(valid_data, batch_size=32)
    fashion_test_loader = DataLoader(test_data, batch_size=32)
    return fashion_train_loader, fashion_valid_loader


@app.cell
def _(train_data):
    fashion_X_sample, fashion_y_sample = train_data[0]
    return


@app.cell
def _(nn):
    # Build the classifer

    class ImageClassifier(nn.Module):
        def __init__(self, n_inputs, n_hidden1, n_hidden2, n_classes):
            super().__init__()
            self.mlp = nn.Sequential(
                nn.Flatten(),
                nn.Linear(n_inputs, n_hidden1),
                nn.ReLU(),
                nn.Linear(n_hidden1, n_hidden2),
                nn.ReLU(),
                nn.Linear(n_hidden2, n_classes)
            )

        def forward(self, X):
            return self.mlp(X)

    return (ImageClassifier,)


@app.cell
def _(device, evaluate_tm):
    def train2(model, optimizer, criterion, metric, train_loader, valid_loader, n_epochs):
        history = {"train_losses": [], "train_metrics": [], "valid_metrics": []}

        for epoch in range(n_epochs):
            total_loss = 0.
            metric.reset()

            for X_batch, y_batch in train_loader:
                model.train()
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                y_pred = model(X_batch)
                loss = criterion(y_pred, y_batch)
                total_loss += loss.item()
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                metric.update(y_pred, y_batch)

            mean_loss = total_loss / len(train_loader)
            history["train_losses"].append(mean_loss)
            history["train_metrics"].append(metric.compute().item())
            history["valid_metrics"].append(evaluate_tm(model, valid_loader, metric).item())
            print(f"Epoch {epoch + 1}/{n_epochs}, "
                  f"train loss: {history['train_losses'][-1]:.4f}, "
                  f"train metric: {history['train_metrics'][-1]:.4f}, "
                  f"valid metric: {history['valid_metrics'][-1]:.4f}")

        return history

    return (train2,)


@app.cell
def _(ImageClassifier, device, nn, torch):
    torch.manual_seed(42)
    class_model = ImageClassifier(n_inputs=28 * 28, n_hidden1=200, n_hidden2=100, n_classes=10).to(device)
    xentropy = nn.CrossEntropyLoss()
    return class_model, xentropy


@app.cell
def _(
    class_model,
    device,
    fashion_train_loader,
    fashion_valid_loader,
    torch,
    torchmetrics,
    train2,
    xentropy,
):
    class_opt = torch.optim.SGD(class_model.parameters(), lr=0.05)
    accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=10).to(device)
    _ = train2(class_model, class_opt, xentropy, accuracy, fashion_train_loader, fashion_valid_loader, 20)
    return


@app.cell
def _(device, fashion_valid_loader, model):
    model.eval()
    fashion_X_new, fashion_y_new = next(iter(fashion_valid_loader))
    fashion_X_new = fashion_X_new[:3].to(device)
    return (fashion_X_new,)


@app.cell
def _(class_model, fashion_X_new, torch):
    with torch.no_grad():
        y_pred_logits = class_model(fashion_X_new)
    return (y_pred_logits,)


@app.cell
def _(y_pred_logits):
    fashion_y_pred = y_pred_logits.argmax(dim=1)
    return (fashion_y_pred,)


@app.cell
def _(fashion_y_pred, train_and_valid_data):
    # fashion_y_pred : [7, 4, 2]

    [train_and_valid_data.classes[index] for index in fashion_y_pred]
    return


@app.cell
def _(y_pred_logits):
    import torch.nn.functional as F
    y_prob = F.softmax(y_pred_logits, dim=1)
    y_prob.round(decimals=3)
    return (F,)


@app.cell
def _(F, torch, y_pred_logits):
    y_top4_logits, y_top4_indices = torch.topk(y_pred_logits, k=4, dim=1)
    y_top4_probs = F.softmax(y_top4_logits, dim=1)
    return y_top4_indices, y_top4_probs


@app.cell
def _(y_top4_probs):
    y_top4_probs.round(decimals=3)
    return


@app.cell
def _(y_top4_indices):
    y_top4_indices
    return


@app.cell
def _(
    ImageClassifier,
    device,
    epoch,
    fashion_train_loader,
    fashion_valid_loader,
    nn,
    torch,
    torchmetrics,
    train2,
):
    # Fine-tune NN hyperparameters with Optuna

    import optuna

    def objective(trial):
        learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-1, log=True)
        n_hidden = trial.suggest_int("n_hidden", 20, 300)
        model = ImageClassifier(n_inputs=1 * 28 * 28, n_hidden1=n_hidden, n_hidden2=n_hidden, n_classes=10).to(device)
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
        xentropy = nn.CrossEntropyLoss()
        accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=10)
        accuracy = accuracy.to(device)
        history = train2(model, optimizer, xentropy, accuracy, fashion_train_loader, fashion_valid_loader, n_epochs=10)
        validation_accuracy = max(history["valid_metrics"])
    
        trial.report(validation_accuracy, epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()
    
        return validation_accuracy    

    return objective, optuna


@app.cell
def _(objective, optuna, torch):
    torch.manual_seed(42)
    sampler = optuna.samplers.TPESampler(seed=42)

    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=0, interval_steps=1)
    study = optuna.create_study(direction="maximize", sampler=sampler, pruner=pruner)
    study.optimize(objective, n_trials=5)
    return


@app.cell
def _(class_model, torch):
    torch.save(class_model, "fashion_mnist.pt")
    return


@app.cell
def _(fashion_X_new, torch):
    loaded_model = torch.load("fashion_mnist.pt", weights_only=False)
    loaded_model.eval()
    loaded_model(fashion_X_new)
    return


@app.cell
def _(class_model, torch):
    torch.save(class_model.state_dict(), "fashion_mnist_weights.pt")
    return


@app.cell
def _():
    # new_model = ImageClassifier(n_inputs=1 * 28 * 28, n_hidden1=200, n_hidden2=100, n_classes=10).to(device)
    # loaded_weights = torch.load("fashion_mnist_weights.pt", weights_only=True)
    # new_model.load_state_dict(loaded_weights)
    # new_model.eval()
    # new_model(fashion_X_new)
    return


@app.cell
def _(class_model, torch):
    model_data = {
        "model_state_dict": class_model.state_dict(),
        "model_hyperparameters": {
            "n_inputs": 1 * 28 * 28,
            "n_hidden1": 200,
            "n_hidden2": 100,
            "n_classes": 10
        }
    }

    torch.save(model_data, "fashion_mnist_model.pt")
    return


@app.cell
def _(ImageClassifier, device, fashion_X_new, torch):
    loaded_data = torch.load("fashion_mnist_model.pt", weights_only=True)
    new_model = ImageClassifier(**loaded_data["model_hyperparameters"]).to(device)
    new_model.load_state_dict(loaded_data["model_state_dict"])
    new_model.eval()
    new_model(fashion_X_new)
    return


@app.cell
def _(torch):
    # Exercise 13

    _x = torch.tensor(1.2, requires_grad=True)
    _y = torch.tensor(3.4, requires_grad=True)

    fxy = torch.sin(_x**2 * _y)
    fxy.backward()

    print(fxy)
    print([_x.grad, _y.grad])

    # tensor(-0.9832, grad_fn=<SinBackward0>)
    # [tensor(1.4899), tensor(0.2629)]
    return


@app.cell
def _(nn):
    # Exercise 14

    class Dense(nn.Module):

        def __init__(self, in_features, out_features):
            super().__init__()
            self.linear = nn.Linear(in_features, out_features)
            self.relu = nn.ReLU()

        def forward(self, X):
            return self.relu(self.linear(X))

    return (Dense,)


@app.cell
def _(Dense, torch):
    torch.manual_seed(42)
    dense = Dense(3, 5)
    _X = torch.randn(2, 3)
    _y_pred = dense(_X)
    _y_pred.shape
    print(_y_pred)
    return


@app.cell
def _(F, nn, torch):
    class Dense2(nn.Module):

        def __init__(self, in_features, out_features):
            super().__init__()
            self.weight = nn.Parameter(torch.randn(out_features, in_features))
            self.bias = nn.Parameter(torch.zeros(out_features))

        def forward(self, X):
            z = X @ self.weight.T + self.bias
            return F.relu(z)

    return (Dense2,)


@app.cell
def _(Dense2, F, torch):
    torch.manual_seed(42)
    dense2 = Dense2(3, 5)
    _X = torch.randn(2, 3)
    y_pred2 = dense2(_X)
    y_pred2.shape

    y_pred2_check = F.relu(_X @ dense2.weight.T + dense2.bias)
    torch.allclose(y_pred2, y_pred2_check)
    return


@app.cell
def _():
    # Exercise 15


    return


if __name__ == "__main__":
    app.run()
