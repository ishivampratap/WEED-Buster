'''
The enable_mixed_precision function is used to enable mixed precision training for the 
model and optimizer, this function is taking the model and optimizer and returning the model 
and optimizer after enabling the mixed precision by using oneAPI's dnnl library's initialize 
function. The accuracy function is used to calculate the accuracy of the model on the provided 
data by using data_loader.
It's important to note that this script is just an example, and the actual utility functions 
may be different depending on the specific use case and requirements.
'''

import oneapi.dnnl as dnnl
import torch

def enable_mixed_precision(model, optimizer):
    """
    Enable mixed precision training for the model
    and optimizer
    """
    model, optimizer = dnnl.amp.initialize(model, optimizer, opt_level='O1')
    return (model, optimizer)

def accuracy(model, data_loader):
    """
    Calculate the accuracy of the model on the provided data
    """
    model.eval()
    correct = 0
    total = 0
    for inputs, labels in data_loader:
        inputs, labels = inputs.cuda(), labels.cuda()
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    return (100 * correct / total)
