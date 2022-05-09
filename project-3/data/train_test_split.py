import pandas as pd
from sklearn.model_selection import train_test_split

filename = 'Skyserver_SQL5_9_2022 6_18_15 PM.csv'

df = pd.read_csv(filename, sep=',')
#df = df[df.iloc[:,-1] < 0.7]
#df = df[df.iloc[:,-1] > 0.4]
df_train, df_test = train_test_split(df, test_size=0.3, random_state=123)

print(df.shape)
print(df_train.shape)
print(df_test.shape)

df_train.to_csv(filename[:-4]+'_train.csv', sep=' ', header=None, index=False)
df_test.to_csv(filename[:-4]+'_test.csv', sep=' ', header=None, index=False)
