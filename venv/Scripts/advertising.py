import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report,confusion_matrix

ad_data = pd.read_csv("C://Programming//Python//Python for DS//13-Logistic-Regression//advertising.csv")




plot1 = plt.figure(1)
sns.set_style('whitegrid')
sns.histplot(data=ad_data,x='Age')

plot2 = plt.figure(2)
sns.jointplot(data=ad_data,x="Age",y="Area Income")
plt.show()

print(ad_data.head())