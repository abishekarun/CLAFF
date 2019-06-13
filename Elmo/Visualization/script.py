import pandas as pd
df=pd.read_csv('agency_yes_rank.csv',sep=',',index_col=0)
#df2=pd.read_csv('social_no_rank.csv',sep=',',index_col=0)
#print(df.shape)
#print(df2.shape)

df=df.drop([0,2])
df=df.iloc[0:34]

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
#print(df['score'].iloc[-1])
plot = plt.scatter(df['score'],df.index, c=df['score'], cmap='Greens')
plt.clf()
plt.colorbar(plot)

ax = sns.barplot(df['score'],df['Word'], palette='Greens_r')
plt.xticks([])
plt.xlabel('')
plt.ylabel('Words')
#plt.show()
plt.savefig('agency_yes_words.png')

