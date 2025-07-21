# train.py

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

import config
from dataloader import EEGDataModule
from model import EEGNet
from engine import train_one_epoch, evaluate


def create_model(train_loader, device):
    """
    Dynamically creates, configures, and moves the model to the correct device.

    This function infers necessary parameters directly from the data loader.

    Args:
        train_loader (DataLoader): The training data loader to infer shapes from.
        device (torch.device): The device to move the model to.

    Returns:
        torch.nn.Module: The initialized EEGNet model.
        int: The number of classes inferred from the data.
    """
    print("Initializing model from data...")
    
    # Get one batch to infer dimensions
    xb, yb = next(iter(train_loader))
    _, FreqBands, Chans, Samples = xb.shape
    
    # Infer number of classes from the batch labels
    # Note: For robustness, it's better to check the full dataset if possible,
    # but for simplicity, we use the batch here as in the original code.
    nb_classes = len(np.unique(yb.cpu().numpy()))
    
    print(f"  - Inferred FreqBands: {FreqBands}, Chans: {Chans}, Samples: {Samples}")
    print(f"  - Inferred Number of Classes: {nb_classes}")

    # Create the model using a mix of inferred params and static configs
    model = EEGNet(
        nb_classes=nb_classes,
        FreqBands=FreqBands,
        Chans=Chans,
        Samples=Samples,
        dropoutRate=config.DROPOUT_RATE,
        kernLength=config.KERNEL_LENGTH,
        F1=config.F1,
        D=config.D,
        F2=config.F2,
        dropoutType=config.DROPOUT_TYPE
    )

    # Move model to the target device
    model.to(device)
    print(f"Model created and moved to {device}.")
    
    return model, nb_classes


if __name__ == '__main__':
    # Setup Device
    device = torch.device(config.DEVICE) if config.DEVICE != "auto" else \
             (torch.device("cuda") if torch.cuda.is_available() else \
             (torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")))
    print(f"Using device: {device}")

    # Prepare Data
    data_module = EEGDataModule(data_dir="/Volumes/Public/data/", batch_size=config.BATCH_SIZE)
    data_module.setup()
    train_loader = data_module.train_dataloader()
    val_loader = data_module.val_dataloader()
    
    # Create Model using our new function
    # The setup logic is now neatly contained within create_model
    model, num_classes = create_model(train_loader, device)

    # Initialize Optimizer and Loss Function
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

    # Training Loop
    print("\n--- Starting Training ---")
    for epoch in range(config.EPOCHS):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        
        print(
            f"Epoch {epoch+1}/{config.EPOCHS} | "
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}"
        )
    print("--- Training Finished ---")
