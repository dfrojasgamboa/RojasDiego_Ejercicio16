import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split # Split arrays or matrices into random train and test subsets
import sklearn.metrics
import sklearn.tree
from pandas.plotting import scatter_matrix

%matplotlib inline

# Carga datos
data = pd.read_csv('OJ.csv')

# Remueve datos que no se van a utilizar
data = data.drop(['Store7', 'PctDiscCH','PctDiscMM'],axis=1)

# Crea un nuevo array que sera el target, 0 si MM, 1 si CH
purchasebin = np.ones(len(data), dtype=int)
ii = np.array(data['Purchase']=='MM')
purchasebin[ii] = 0

data['Target'] = purchasebin

# Borra la columna Purchase
data = data.drop(['Purchase'],axis=1)

# Crea un dataframe con los predictores
predictors = list(data.keys())
predictors.remove('Target')
predictors.remove('Unnamed: 0')
print(predictors)