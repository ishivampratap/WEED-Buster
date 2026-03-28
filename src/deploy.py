
'''
The above script will be used to deploy the model in a simulated production environment, 
the model is set to evaluation mode so that the dropout will be turned off. and the test 
dataset is passed through the model, the model's predictions are compared with the actual 
labels and the accuracy of the model is calculated. But this time i am using oneAPI's dnnl 
library's accuracy function to calculate the accuracy of the model on the test set.
It's important to note that this script is just an example, and the actual deployment 
process may be different depending on the specific use case and requirements.
'''

import oneapi

def deploy_model(model, test_loader):
    """
    Deploy the model in a simulated production environment
    and evaluate the model's performance on the test set
    """
    model.eval()
    with torch.no_grad():
        accuracy = dnnl.accuracy(model, test_loader)
    print(f'Accuracy of the model on the test set: {accuracy} %')


