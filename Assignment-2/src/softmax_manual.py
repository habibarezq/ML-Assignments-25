from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import torch
from torch.utils.data import TensorDataset,DataLoader ## to feed the data into the model
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import torch
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

class SoftmaxRegression:
  def __init__(self, input_dim, num_classes, learning_rate=0.01, patience=10, min_delta=0.0005, max_epochs=200, early_stopping=True):
    self.input_dim = input_dim
    self.num_classes = num_classes
    self.learning_rate = learning_rate
    # self.reg_lambda = reg_lambda
    self.patience = patience
    self.min_delta = min_delta
    self.max_epochs = max_epochs
    self.early_stopping = early_stopping

    # Initialize weights with small random values
    self.W = torch.randn(self.input_dim, self.num_classes) * 0.01
    self.b = torch.zeros(self.num_classes) #Compute the grad manually

    # Store training history
    self.history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }


  def softmax(self, z):
    exp_z = torch.exp(z - torch.max(z, dim=1, keepdim=True)[0])  # stability trick
    return exp_z / torch.sum(exp_z, dim=1, keepdim=True)


  def cross_entropy_loss(self,y_pred,y_true):
    eps=1e-8
    y_true_one_hot = torch.nn.functional.one_hot(y_true.long(), num_classes=self.num_classes).float()
    loss= -torch.mean(torch.sum(y_true_one_hot * torch.log(y_pred + eps), dim=1))
    return loss

  def forward_pass(self, X_batch):
      logits = X_batch @ self.W + self.b
      y_pred = self.softmax(logits)
      return y_pred

  """### Accuracy and Gradients"""
  def compute_accuracy(self, y_pred, y_true):
      preds = torch.argmax(y_pred, dim=1)
      accuracy = (preds == y_true).float().mean()
      return accuracy.item()

  def compute_gradients(self, X_batch, y_batch, y_pred):
      y_true_onehot = torch.nn.functional.one_hot(y_batch.long(), num_classes=y_pred.shape[1]).float()
      error = (y_pred - y_true_onehot) / X_batch.shape[0]
      dW = X_batch.T @ error
      db = error.sum(dim=0)
      return dW, db

  def update_weights(self, dW, db):
      self.W -= self.learning_rate * dW
      self.b -= self.learning_rate * db

  def fit(self, train_loader, val_loader):
      train_losses, val_losses = [], []
      train_accuracies, val_accuracies = [], []

      best_val_loss = float('inf')
      epochs_without_improvement = 0
      best_W, best_b = None, None

      for epoch in range(self.max_epochs):
          train_loss_epoch, train_acc_epoch = 0, 0
          n_train_batches = 0

          # Training Phase
          for X_batch, y_batch in train_loader:
              y_pred = self.forward_pass(X_batch)
              loss = self.cross_entropy_loss(y_pred, y_batch)
              dW, db = self.compute_gradients(X_batch, y_batch, y_pred)
              self.update_weights(dW, db)

              train_loss_epoch += loss.item()
              train_acc_epoch += self.compute_accuracy(y_pred, y_batch)
              n_train_batches += 1

          # Validation Phase
          val_loss_epoch, val_acc_epoch, n_val_batches = 0, 0, 0
          with torch.no_grad():
              for X_batch, y_batch in val_loader:
                  y_pred = self.forward_pass(X_batch)
                  loss = self.cross_entropy_loss(y_pred, y_batch)
                  val_loss_epoch += loss.item()
                  val_acc_epoch += self.compute_accuracy(y_pred, y_batch)
                  n_val_batches += 1

          # Record each epoch metrics
          avg_train_loss = train_loss_epoch / n_train_batches
          avg_val_loss = val_loss_epoch / n_val_batches

          train_losses.append(avg_train_loss)
          train_accuracies.append(train_acc_epoch / n_train_batches)
          val_losses.append(avg_val_loss)
          val_accuracies.append(val_acc_epoch / n_val_batches)

          # Print progress
          if (epoch + 1) % 10 == 0:
              print(f"Epoch {epoch+1}/{self.max_epochs} - "
                    f"Train Loss: {train_losses[-1]:.4f}, Train Acc: {train_accuracies[-1]:.4f} - "
                    f"Val Loss: {val_losses[-1]:.4f}, Val Acc: {val_accuracies[-1]:.4f}")

          # Early Stopping logic
          if self.early_stopping:
              if avg_val_loss < best_val_loss - self.min_delta:
                  best_val_loss = avg_val_loss
                  epochs_without_improvement = 0
                  best_W = self.W.clone()
                  best_b = self.b.clone() # pyright: ignore[reportOptionalMemberAccess]
              else:
                  epochs_without_improvement += 1

              if epochs_without_improvement >= self.patience:
                  print("\n" + "=" * 60)
                  print(f"Early stopping triggered at epoch {epoch+1}")
                  print(f"Best validation loss: {best_val_loss:.4f} at epoch {epoch+1 - self.patience}")
                  print("=" * 60)
                  break

      # Restore best model
      if self.early_stopping and best_W is not None:
          self.W, self.b = best_W, best_b
          print(f"\nRestored best model (Val Loss = {best_val_loss:.4f})")

      # Save training history
      self.history['train_loss'] = train_losses
      self.history['val_loss'] = val_losses
      self.history['train_acc'] = train_accuracies
      self.history['val_acc'] = val_accuracies

      # return self.W, self.b, train_losses, val_losses, train_accuracies, val_accuracies

  """### Visualization"""
  def plot_curves(self):
      """1. Loss Curves"""
      plt.figure(figsize=(10, 6))
      plt.plot(self.history['train_loss'], label='Training Loss', linewidth=2)
      plt.plot(self.history['val_loss'], label='Validation Loss', linewidth=2)
      plt.xlabel('Epoch', fontsize=12)
      plt.ylabel('Loss', fontsize=12)
      plt.title('Softmax Regression: Loss Curves', fontsize=14, fontweight='bold')
      plt.legend()
      plt.grid(True, alpha=0.3)
      plt.show()

      """2. Accuracies Curve"""
      plt.figure(figsize=(10, 6))
      plt.plot(self.history['train_acc'], label='Training Accuracy', linewidth=2)
      plt.plot(self.history['val_acc'], label='Validation Accuracy', linewidth=2)
      plt.xlabel('Epoch', fontsize=12)
      plt.ylabel('Accuracy', fontsize=12)
      plt.title('Softmax Regression: Accuracy Curves', fontsize=14, fontweight='bold')
      plt.legend()
      plt.grid(True, alpha=0.3)
      plt.show()

  """### Test data"""
  def evaluate_test(self, X_test, y_test):
      with torch.no_grad():
          y_pred_test = self.forward_pass(X_test)
          test_predictions = torch.argmax(y_pred_test, dim=1)
          test_accuracy = self.compute_accuracy(y_pred_test, y_test)
          test_loss = self.cross_entropy_loss(y_pred_test, y_test)

      print(f"Test Loss: {test_loss:.4f}")
      print(f"Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
      return test_predictions, y_pred_test

  """### Confusion Matrix"""
  def plot_confusion_matrix(self, y_test, test_predictions, class_labels=None):
      y_test_np = y_test.numpy()
      test_predictions_np = test_predictions.numpy()
      cm = confusion_matrix(y_test_np, test_predictions_np)

      print("Confusion Matrix")
      print(cm)

      plt.figure(figsize=(8, 6))
      sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                  xticklabels=class_labels if class_labels else range(cm.shape[0]), # pyright: ignore[reportArgumentType]
                  yticklabels=class_labels if class_labels else range(cm.shape[0])) # pyright: ignore[reportArgumentType]
      plt.title('Confusion Matrix - Softmax Regression', fontsize=14, fontweight='bold')
      plt.ylabel('True Label', fontsize=12)
      plt.xlabel('Predicted Label', fontsize=12)
      plt.tight_layout()
      plt.show()

