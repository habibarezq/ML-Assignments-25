import torch
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split

class MNISTDataLoader:
    def __init__(self, batch_size=64, flatten=True, binary=False, digits=(0, 1), seed=42):
        self.batch_size = batch_size
        self.flatten = flatten
        self.binary = binary
        self.digits = digits
        self.seed = seed
        
        self._prepare_data()

    def _prepare_data(self):
        transform = transforms.ToTensor()

        # Download and load MNIST dataset
        mnist_train = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        mnist_test = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

        # Combine train+test for stratified splitting
        X_full = torch.cat([mnist_train.data, mnist_test.data], dim=0).float() / 255.0
        y_full = torch.cat([mnist_train.targets, mnist_test.targets], dim=0)

        # If binary, filter only selected digits
        if self.binary:
            mask = (y_full == self.digits[0]) | (y_full == self.digits[1])
            X_full = X_full[mask]
            y_full = y_full[mask]

        # Split data (60% train, 20% val, 20% test)
        X_temp, X_test, y_temp, y_test = train_test_split(
            X_full, y_full, test_size=0.20, stratify=y_full, random_state=self.seed
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=0.25, stratify=y_temp, random_state=self.seed
        )

        # Flatten images if needed
        if self.flatten:
            X_train = X_train.reshape(-1, 28*28)
            X_val = X_val.reshape(-1, 28*28)
            X_test = X_test.reshape(-1, 28*28)
        else:
            # add a channel dimension for cnn
            X_train = X_train.unsqueeze(1)  
            X_val = X_val.unsqueeze(1)     
            X_test = X_test.unsqueeze(1)

        # Store splits
        self.X_train, self.X_val, self.X_test = X_train, X_val, X_test
        self.y_train, self.y_val, self.y_test = y_train, y_val, y_test

    def get_loaders(self):
        # Ensure labels are float only if binary classification
        label_type = torch.float32 if self.binary else torch.long

        train_loader = DataLoader(
            TensorDataset(self.X_train, self.y_train.to(label_type)),
            batch_size=self.batch_size, shuffle=True
        )
        val_loader = DataLoader(
            TensorDataset(self.X_val, self.y_val.to(label_type)),
            batch_size=self.batch_size, shuffle=False
        )
        test_loader = DataLoader(
            TensorDataset(self.X_test, self.y_test.to(label_type)),
            batch_size=self.batch_size, shuffle=False
        )

        return train_loader, val_loader, test_loader
