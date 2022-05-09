import pandas as pd
from sklearn.model_selection import train_test_split

filename = 'sky_data_model.txt'

df = pd.read_csv(filename, header=None, sep=' ')
df_train, df_test = train_test_split(df, test_size=0.2, random_state=123)

print(df.shape)
print(df_train.shape)
print(df_test.shape)

df_train.to_csv(filename[:-4]+'_train.txt', sep=' ', header=None, index=False)
df_test.to_csv(filename[:-4]+'_test.txt', sep=' ', header=None, index=False)
