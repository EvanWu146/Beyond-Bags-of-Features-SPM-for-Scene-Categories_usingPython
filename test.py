import pickle
import numpy as np
from sklearn.metrics import confusion_matrix

f = open('predict', 'rb')
d = pickle.load(f)
print("Accuracy:", np.mean(d['predict'] == d['label']) * 100, "%")
m = confusion_matrix(d['label'], d['predict'])
print(m)
