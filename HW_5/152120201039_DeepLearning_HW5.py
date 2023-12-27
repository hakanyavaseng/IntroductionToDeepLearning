if 1:
    from google.colab import drive
    drive.mount('/content/drive')
     
     

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Loading data from Google Drive
X_train = np.load('/content/drive/MyDrive/Deep_Learning/data/DataForClassification_TimeDomain.npy')
X_train = np.transpose(X_train)

# Generating labels
labels = np.zeros((936, 1))
label = 0
for i in range(936):
    labels[i] = label
    if (i % 104 == 0) and (i != 0):
        label = label + 1

# Encoding labels using OneHotEncoder
encoder = OneHotEncoder(sparse_output=False)
Y_train = encoder.fit_transform(labels)

# Train-Test Split
X_dev, X_test, Y_dev, Y_test = train_test_split(X_train, Y_train, test_size=0.2, shuffle=True, random_state=0)
X_train, X_val, Y_train, Y_val = train_test_split(X_dev, Y_dev, test_size=0.2, shuffle=True, random_state=0)

# Converting data to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
X_val = torch.tensor(X_val, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
Y_train = torch.tensor(Y_train, dtype=torch.float32)
Y_val = torch.tensor(Y_val, dtype=torch.float32)
Y_test = torch.tensor(Y_test, dtype=torch.float32)

# Expanding dimensions
X_train = X_train.unsqueeze(1)
X_val = X_val.unsqueeze(1)
X_test = X_test.unsqueeze(1)

# Defining model
class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(GRUModel, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)
        self.fc1 = nn.Linear(hidden_size, 8)
        self.fc2 = nn.Linear(8, num_classes)

    def forward(self, x):
        out, _ = self.gru(x)
        out = self.fc1(out[:, -1, :])
        out = self.fc2(out)
        return out

# Initializing model
model = GRUModel(input_size=3600, hidden_size=32, num_classes=9)

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3) #Used Adam like in Keras code

# Learning rate scheduler
scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.3, patience=7, min_lr=1e-5, verbose=True)

# Training loop
num_epochs = 50
batch_size = 64

train_dataset = TensorDataset(X_train, torch.argmax(Y_train, dim=1))
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

for epoch in range(num_epochs):
    model.train()
    for inputs, labels in train_loader:
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Validation
    model.eval()
    with torch.no_grad():
        val_outputs = model(X_val)
        val_loss = criterion(val_outputs, torch.argmax(Y_val, dim=1))
        scheduler.step(val_loss)

    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}, Validation Loss: {val_loss.item()}')

# Saving the model
torch.save(model.state_dict(), '/content/nn_model.pth')

# Testing the model
model.eval()
with torch.no_grad():
    test_outputs = model(X_test)
    test_predictions = torch.argmax(test_outputs, dim=1)

# Converting predictions to NumPy array
test_predictions_np = test_predictions.numpy()
print(test_predictions_np)


# Doing this to see the accuracy on the test set
# Converting predictions to NumPy array
test_predictions_np = test_predictions.numpy()

# Converting ground truth labels to NumPy array
Y_test_np = torch.argmax(Y_test, dim=1).numpy()

# Comparing predictions with ground truth
correct_predictions = (test_predictions_np == Y_test_np)
accuracy = correct_predictions.sum() / len(correct_predictions)


print(f"\n\nAccuracy on the test set: {accuracy * 100:.2f}%")