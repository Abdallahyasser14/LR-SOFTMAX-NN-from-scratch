import torch
from torchvision import datasets, transforms


def filter_two_classes(X, Y):
    mask = (Y == 1) | (Y == 0)
    X_filtered = X[mask]
    Y_filtered = Y[mask]
    
    Y_binary = (Y_filtered == 1).float()  
    return X_filtered, Y_binary


def load_lr_data() :
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


    # --- Training ---
    X_train = train_dataset.data.float() / 255.0
    X_train = X_train.reshape(-1, 28 * 28) 
    Y_train = train_dataset.targets       

    # --- Test ---
    X_test = test_dataset.data.float() / 255.0
    X_test = X_test.reshape(-1, 28 * 28) 
    Y_test = test_dataset.targets     

    print(f"X_train shape: {X_train.shape}")
    print(f"Y_train shape: {Y_train.shape}")





    X_train_bin, Y_train_bin = filter_two_classes(X_train, Y_train)
    X_test_bin,  Y_test_bin  = filter_two_classes(X_test,  Y_test)

    Y_train_bin = Y_train_bin.unsqueeze(1)
    Y_test_bin  = Y_test_bin.unsqueeze(1)
    return  X_train_bin, Y_train_bin,X_test_bin,Y_test_bin


