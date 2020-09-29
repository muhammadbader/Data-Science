import pandas as pd
import numpy as np
from numpy.random import randn
import matplotlib.pyplot as plt
import seaborn as sns


##### tests = []
# print(rd.Random(4000))

p = pd.DataFrame(randn(3, 4), columns="H M S, total_tests".split())

print(p)
# herer now
fig = plt.figure()
axis = fig.add_axes([0.1, 0.1, 0.8, 0.8])

sns.lmplot(x='total_tests', y='M', data=p)
plt.show()

