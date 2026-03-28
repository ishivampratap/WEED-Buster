
'''
The above script will load the dataset and perform the Exploratory Data Analysis on 
the provided dataframe using oneAPI and PyTorch library along with Anaconda library. 
The script will provide a summary of the data, will show the count of weeds in the data, 
will show the pairplot of the data, and will show the heatmap of the data.It could be run 
in Jupyter notebook and it will show the output in the notebook.
It's important to note that this script is just an example, and the actual EDA process may 
be different depending on the specific use case and requirements.
'''

# Importing the libraries
import oneapi
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv("weeds_data.csv")

# Perform Exploratory Data Analysis
print(df.info())
print(df.describe())
sns.countplot(x='Weeds', data=df)
plt.show()
sns.pairplot(df, hue='Weeds')
plt.show()
sns.heatmap(df.corr(), annot=True)
plt.show()
