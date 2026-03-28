'''
In this script, I'm using the oneAPI Dataloader to read and preprocess the dataset and then 
i am splitting the dataset into train, validation, and test sets. And then returning the 
dataloader objects.
It is important to note that oneAPI DAL is a library that provides a set of data loading 
and preprocessing functions that can be used with deep learning frameworks such as PyTorch 
and TensorFlow, so it can be used with this script as well.
'''

import torch
import oneapi
from oneapi.dataloader import DataLoader

def preprocess_data(data_path, batch_size=32):
    """
    Reads images and labels from the data path, and preprocess them using oneAPI DAL
    Returns:
        train_loader, val_loader, test_loader
    """
    # Create a DataLoader object
    data_loader = DataLoader(data_path)
    
    # Use oneAPI DAL to preprocess the data
    data_loader.normalize()
    data_loader.shuffle()
    data_loader.split(0.8, 0.1, 0.1)
    data_loader.to_tensor()

    # Create DataLoaders for train, validation, and test sets
    train_loader = torch.utils.data.DataLoader(data_loader.train_data, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(data_loader.val_data, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(data_loader.test_data, batch_size=batch_size, shuffle=True)
    
    return train_loader, val_loader, test_loader


