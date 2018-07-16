import numpy as np
import pandas as pd
from plsa2 import PLSA

N = pd.read_csv('plsa_dataset.csv',index_col=0)
# 数字はz
plsa = PLSA(N, 8)
plsa.train()

print ('P(z)')
# print (plsa.Pz)
pd.DataFrame(plsa.Pz).to_csv("Pz.csv", header=None, index=None)
print ('P(d|z)')
# print (plsa.Px_z)
pd.DataFrame(plsa.Px_z).to_csv("Pd_z.csv", header=None, index=None)
print ('P(w|z)')
# print (plsa.Py_z)
pd.DataFrame(plsa.Py_z).to_csv("Pw_z.csv", header=None, index=None)
print ('P(w|d)')
Pz_x = plsa.Px_z.T * plsa.Pz[None, :]
# print (Pz_x / np.sum(Pz_x, axis=1)[:, None])
print ('P(z|w)')
Pz_y = plsa.Py_z.T * plsa.Pz[None, :]
# print (Pz_y / np.sum(Pz_y, axis=1)[:, None])
