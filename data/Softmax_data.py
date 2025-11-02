import torch
from torchvision import datasets, transforms

def one_hot_encode(y, num_classes=10):
    return torch.eye(num_classes)[y]

def load_softmax_data():

    train_dataset = datasets.MNIST(
        root='./MNIST_DATASET',
        train=True,
        download=True,
        transform=transforms.ToTensor()
    )

    test_dataset = datasets.MNIST(
        root='./MNIST_DATASET',
        train=False,
        download=True,
        transform=transforms.ToTensor()
    )


    # --- Training Data ---
    X_train = train_dataset.data.float() / 255.0
    X_train = X_train.reshape(-1, 28 * 28)
    Y_train = train_dataset.targets     

    # --- Test Data ---
    X_test = test_dataset.data.float() / 255.0
    X_test = X_test.reshape(-1, 28 * 28) 
    Y_test = test_dataset.targets     

    print(f"X_train shape: {X_train.shape}")
    print(f"Y_train shape: {Y_train.shape}")


    Y_train_onehot = one_hot_encode(Y_train, num_classes=10)
    Y_test_onehot = one_hot_encode(Y_test, num_classes=10)

    

    return X_train, Y_train_onehot, X_test, Y_test_onehot