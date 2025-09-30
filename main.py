import torch
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split


n_samples = 1400

X, y = make_circles(n_samples,
                    noise=0.03,
                    random_state=42)
## print(f"{len(X)} | {len(y)}") "Used for testing and exploring the data"

circles =pd.DataFrame({"X1" : X[:, 0],
                       "X2" : X[:, 1],
                       "label": y})
print(circles.head(10)) #"Used for testing and exploring the data"

plt.scatter(x=X[:, 0],
            y=X[:, 1],
            c=y,
            cmap=plt.cm.RdYlBu)
#plt.show() "Used for testing and exploring the data"

# Turning data into tensors & create test and train splits

##print(f"{X.shape, y.shape, type(X), X.dtype}")"Used for testing the data"

X = torch.from_numpy(X).type(torch.float)
y= torch.from_numpy(y).type(torch.float)

##print(f"{X.shape, y.shape, type(X), X.dtype}\n{X[:5], y[:5]}")"Used for testing the data"

X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.2,
                                                    random_state=42)
##print(f"{len(X_train), len(X_test), len(y_train), len(y_test)}") "Used for testing the data"

# writing device-agnostic code

device = "cuda" if torch.cuda.is_available() else "cpu"

# Instantiating a model
from model import CircleModel_v0
model_1 =CircleModel_v0().to(device)


#print(model_1.state_dict()) "Used for testing the data"
# print({X_test[:10], y_test[:10]})"Used for testing the data"


#Choosing a loss function and optimizer

loss_fn = torch.nn.BCEWithLogitsLoss()

optimizer = torch.optim.SGD(params=model_1.parameters(),
                            lr=0.1)

torch.manual_seed(42)
torch.cuda.manual_seed(42)
epochs = 1600

X_train, y_train = X_train.to(device), y_train.to(device)
X_test, y_test = X_test.to(device), y_test.to(device)

from helper_fn import accuracy_fn
train_losses, test_losses = [], []
train_accs, test_accs = [], []


for epoch in range(epochs):
    model_1.train()
    #forward pass
    y_logits = model_1(X_train).squeeze()
    y_preds = torch.round(torch.sigmoid(y_logits))
    #calculate loss/accuracy
    loss = loss_fn(y_logits,
                   y_train)

    acc = accuracy_fn(y_train,
                      y_preds)

    #Backprobagation
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


    model_1.eval()
    with torch.inference_mode():
        test_logits = model_1(X_test).squeeze()
        test_preds = torch.round(torch.sigmoid(test_logits))

        test_loss =loss_fn(test_preds,
                           y_test)
        test_acc = accuracy_fn(y_test,
                               test_preds)

        train_losses.append(loss.item())
        test_losses.append(test_loss.item())
        train_accs.append(acc)
        test_accs.append(test_acc)

        if epoch %200 == 0:
            print(f"Epoch:{epoch} | Loss:{loss:.5f} | Acc:{acc:.2f}% | Teset Loss: {test_loss:.5f} | Test Acc{test_acc:.2f} ")




from helper_fn import plot_decision_boundary, accuracy_fn

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title("Train")
plot_decision_boundary(model_1, X_train, y_train)
plt.subplot(1, 2, 2)
plt.title("Test")
plot_decision_boundary(model_1, X_test, y_test)



# Plot loss & accuracy curves in one figure
plt.figure(figsize=(12, 5))

# Loss subplot
plt.subplot(1, 2, 1)
plt.plot(train_losses, label="Train Loss")
plt.plot(test_losses, label="Test Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training and Test Loss")
plt.legend()

# Accuracy subplot
plt.subplot(1, 2, 2)
plt.plot(train_accs, label="Train Accuracy")
plt.plot(test_accs, label="Test Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy (%)")
plt.title("Training and Test Accuracy")
plt.legend()

plt.tight_layout()
plt.show()
