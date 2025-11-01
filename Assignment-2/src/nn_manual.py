import numpy as np
import torch
import torch.nn as nn           # imports common layers
import torch.nn.functional as F # provides functional versions of activations like ReLU
import torch.optim as optim     # provides optimization algorithms like SGD
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

class CustomFeedforwardNN(nn.Module):
  # 1st hidden layer: 128 neurons, 2nd hidden layer: 64 neurons
    def __init__(self, input_size=784, hidden1_size=128, hidden2_size=64, output_size=10):

        super(CustomFeedforwardNN, self).__init__()

        self.fc1 = nn.Linear(input_size, hidden1_size)
        self.fc2 = nn.Linear(hidden1_size, hidden2_size)
        self.fc3 = nn.Linear(hidden2_size, output_size)

        # initialize weights with Xavier initialization (helps keep gradients stable)
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.fc3.weight)

        # initialize biases to zero
        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.bias)
        nn.init.zeros_(self.fc3.bias)

    def forward(self, x): # x is input tensor of size (batch size, 784)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x  # output tensor of size (batch size, 10)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_model_once(model, train_loader, val_loader, epochs=10, learning_rate=0.01, device=device):
    model.to(device) # transfer to chosen device
    criterion = nn.CrossEntropyLoss()  # cross-entropy
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)  # SGD is used to optimize parameters
    # store metrics for each epoch
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    for epoch in range(epochs):
        # training mode (enable features like dropout)
        model.train()
        # initialize accumulators for batch loss, correct predictions and total samples processed
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device) # move batch inputs and labels to device
            optimizer.zero_grad()   # reset
            outputs = model(inputs) # forward pass
            loss = criterion(outputs, labels) # calculate loss comparing outputs to labels
            loss.backward()         # compute gradients via backpropagation
            optimizer.step()        # update model weights
            running_loss += loss.item() * inputs.size(0) # accumulate total loss (weighted by batch size)
            _, predicted = torch.max(outputs, 1) # calculate predicted class labels
            total += labels.size(0) # total samples processed
            correct += (predicted == labels).sum().item() # calculate correct predictions
        train_loss = running_loss / total
        train_acc = correct / total
        # validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        val_loss /= val_total
        val_acc = val_correct / val_total
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)
        print(f"Epoch {epoch+1}/{epochs} "
              f"Train Loss: {train_loss:.4f} Train Acc: {train_acc:.4f} "
              f"Val Loss: {val_loss:.4f} Val Acc: {val_acc:.4f}")
    return train_losses, val_losses, train_accuracies, val_accuracies

# run the training process multiple times according to runs variable
def train_multiple_times(model_class, train_loader, val_loader, epochs=10, learning_rate=0.01, runs=5):
    all_train_losses = []
    all_val_losses = []
    all_train_accuracies = []
    all_val_accuracies = []

    for r in range(runs):
        print(f"\nRun {r+1}/{runs}")
        # independent model for each training run
        model = model_class()
        # collect training and validation losses and accuracies each run
        train_losses, val_losses, train_accuracies, val_accuracies = train_model_once(
            model, train_loader, val_loader, epochs, learning_rate, device)
        all_train_losses.append(train_losses)
        all_val_losses.append(val_losses)
        all_train_accuracies.append(train_accuracies)
        all_val_accuracies.append(val_accuracies)

    # convert to numpy arrays for easier calculations
    all_train_losses = np.array(all_train_losses)
    all_val_losses = np.array(all_val_losses)
    all_train_accuracies = np.array(all_train_accuracies)
    all_val_accuracies = np.array(all_val_accuracies)

    # mean and std deviation for error bars
    mean_train_loss = np.mean(all_train_losses, axis=0)
    std_train_loss = np.std(all_train_losses, axis=0)
    mean_val_loss = np.mean(all_val_losses, axis=0)
    std_val_loss = np.std(all_val_losses, axis=0)
    mean_train_acc = np.mean(all_train_accuracies, axis=0)
    std_train_acc = np.std(all_train_accuracies, axis=0)
    mean_val_acc = np.mean(all_val_accuracies, axis=0)
    std_val_acc = np.std(all_val_accuracies, axis=0)

    # plot loss with error bars
    epochs_range = range(1, epochs + 1) # used for x-axis
    plt.figure(figsize=(10,5))
    plt.errorbar(epochs_range, mean_train_loss, yerr=std_train_loss, label='Training Loss')
    plt.errorbar(epochs_range, mean_val_loss, yerr=std_val_loss, label='Validation Loss')
    plt.title("Loss over Epochs with Error Bars")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # plot accuracy with error bars
    plt.figure(figsize=(10,5))
    plt.errorbar(epochs_range, mean_train_acc, yerr=std_train_acc, label='Training Accuracy')
    plt.errorbar(epochs_range, mean_val_acc, yerr=std_val_acc, label='Validation Accuracy')
    plt.title("Accuracy over Epochs with Error Bars")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # convergence analysis: find epoch where validation loss change is minimal
    val_loss_diff = np.abs(np.diff(mean_val_loss))
    convergence_epoch = np.argmin(val_loss_diff) + 1
    print(f"Convergence epoch: {convergence_epoch}")


## Part D2 (NN With dropout layers)
class NNDropout(nn.Module):
    def __init__(self,dropout_rate=0.5):
        super().__init__()
        # First Hidden layer
        self.fc1 = nn.Linear(28*28,256)
        self.dropout1=nn.Dropout(dropout_rate)
        # 2nd hidden layer
        self.fc2=nn.Linear(256,128)
        self.dropout2=nn.Dropout(dropout_rate)
        #output layer
        self.fc3=nn.Linear(128,10)
        
    def forward(self,x): # x should be already flattened
        x = F.relu(self.fc1(x)) # fc -> relu -> dropout
        x = self.dropout1(x)
        
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        
        x = self.fc3(x)
        return x
    
## Part D2: NN with Batch Normalization
class NNBatchNormalized(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.fc1=nn.Linear(28*28,256)
        self.bn1=nn.BatchNorm1d(256)
        
        self.fc2=nn.Linear(256,128)
        self.bn2=nn.BatchNorm1d(128)
        
        self.fc3=nn.Linear(128,10)
    
    def forward(self,x):
        x= self.bn1(F.relu(self.fc1(x)))
        x= self.bn2(F.relu(self.fc2(x)))
        x=self.fc3(x)
        return x
    
# NN with dropout and batch normalization (i just combined both)
class NNDropoutBatchNormalized(nn.Module):
    def __init__(self, dropout_rate=0.3):
        super().__init__()
        self.fc1 = nn.Linear(28*28, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.dropout1 = nn.Dropout(dropout_rate)
        
        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.dropout2 = nn.Dropout(dropout_rate)
        
        self.fc3 = nn.Linear(128, 10)
    
    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.bn1(x)
        x = self.dropout1(x)
        
        x = F.relu(self.fc2(x))
        x = self.bn2(x)
        x = self.dropout2(x)
        
        x = self.fc3(x)
        return x
        