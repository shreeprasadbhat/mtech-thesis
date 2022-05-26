import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

filename ='specPhotoDR17Type3.csv' #'MyTable_shreeprasadbhat.csv'

df = pd.read_csv(filename, sep=',')

#df = df
#df = df[df['zWarning'] == 0]
#df = df[df['clean']==1]
#df = df.drop(['clean', 'zWarning'], axis=1)
#df = df[df['u'] > 0]
#df = df[df['g'] > 0]
#df = df[df['r'] > 0]
#df = df[df['i'] > 0]
#df = df[df['z'] > 0]
#error_cut = 2.
#df = df[df['u_err'] < error_cut]
#df = df[df['g_err'] < error_cut]
#df = df[df['r_err'] < error_cut]
#df = df[df['i_err'] < error_cut]
#df = df[df['z_err'] < error_cut]
#df = df[df['specz_err'] < 0.01]
#df = df[(df['specz'] > 0) & (df['specz'] < 1.)]
#
## oversampling
#df = pd.concat([df, df[df['specz'] > 0.7]])
#df = pd.concat([df, df[df['specz'] > 0.75]])
#df = pd.concat([df, df[df['specz'] > 0.8]])
#df = pd.concat([df, df[df['specz'] > 0.9]])
#df = pd.concat([df, df[df['specz'] > 0.95]])

#dfrain = pd.DataFrame(columns=['u','g','r','i','z','u_err','g_err','r_err','i_err','z_err','specz','specz_err','photoz','photoz_err'])
#dfest = pd.DataFrame(columns=['u','g','r','i','z','u_err','g_err','r_err','i_err','z_err','specz','specz_err','photoz','photoz_err'])
#
#sample_size = 2000
#interval_size = .1
#z_max_train = 1.
#z_max = df['specz'].max()
#
#z_list = np.arange(0, z_max_train, interval_size)
#
#for z in z_list :
#    dfcur = df[(df['specz'] >= z) & (df['specz'] < z+interval_size)]
#    train_size = min(sample_size, int(0.8 * dfcur.shape[0]))
#    dfrain_cur, dfest_cur = train_test_split(dfcur, train_size=train_size, random_state=123)
#    dfrain = pd.concat([dfrain, dfrain_cur])
#    dfest = pd.concat([dfest, dfest_cur])
#
##dfest_full = pd.concat([dfest, df[df['specz']>=1.]])

df = df
df = df[df['zWarning'] == 0]
df = df[df['clean']==1]
df = df.drop(['clean', 'zWarning'], axis=1)
df = df[df['u'] > 0]
df = df[df['g'] > 0]
df = df[df['r'] > 0]
df = df[df['i'] > 0]
df = df[df['z'] > 0]
error_cut = 2. 
df = df[df['u_err'] < error_cut]
df = df[df['g_err'] < error_cut]
df = df[df['r_err'] < error_cut]
df = df[df['i_err'] < error_cut]
df = df[df['z_err'] < error_cut]
df = df[df['specz_err'] < 0.01]
df = df[(df['specz'] > 0) & (df['specz'] < 1.)]

df_new = pd.DataFrame(columns=['u','g','r','i','z','u_err','g_err','r_err','i_err','z_err','specz','specz_err','photoz','photoz_err'])

sample_size = 1000
interval_size = 0.001
z_max_train = 1 
z_max = df['specz'].max()

z_list = np.arange(0, z_max_train, interval_size)

for z in z_list :
    df_cur = df[(df['specz'] >= z) & (df['specz'] < z+interval_size)]
    train_size = min(sample_size, df_cur.shape[0])
    if train_size == 0 : continue
    df_train_cur = df_cur.sample(n=train_size, random_state=123)
    df_new = pd.concat([df_new, df_train_cur])
df = df_new

print(df.shape)

filename = 'MyTable_shreeprasadbhat.csv'

df_train, df_test = train_test_split(df, train_size = 0.8, random_state=123)
#df_test = df_test[df_test['specz'] > 1.]
df_train.to_csv(filename[:-4]+'_train.csv', sep=' ', header=None, index=False)
df_test.to_csv(filename[:-4]+'_test.csv', sep=' ', header=None, index=False)

print(df_train.shape)
print(df_test.shape)
#print(dfest_full.shape)

#dfest_full.to_csv(filename[:-4]+'_test_full.csv', sep=' ', header=None, index=False)
