# config.py

# -- 1. Training Parameters --
LEARNING_RATE = 0.001
BATCH_SIZE = 64
EPOCHS = 100
DEVICE = "auto"  # Options: "auto", "cuda", "mps", "cpu"

# -- 2. Data Preprocessing Parameters --
VARIANCE_TO_KEEP = 0.95 

# -- 3. EEGNet Model Hyperparameters --
DROPOUT_RATE = 0.3
F1 = 8          
D = 2           
F2 = 16         
KERNEL_LENGTH = 64
DROPOUT_TYPE = 'Dropout'