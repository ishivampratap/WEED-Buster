'''
In this script, the model is set to evaluation mode so that the dropout will be turned off. 
and the test dataset is passed through the model, the model's predictions are compared with 
the actual labels and the accuracy of the model is calculated using oneAPI's dnnl library's 
accuracy function.
It's important to note that this script is just an example, and the actual evaluation process 
may be different depending on the specific use case and requirements.
'''

import oneapi

def evaluate_model(model, test_loader):
    """
    Evaluate the model's performance on the test set
    """
    model.eval()
    accuracy = dnnl.accuracy(model, test_loader)
    print(f'Accuracy of the model on the test set: {accuracy} %')

