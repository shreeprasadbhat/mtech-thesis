import os
import yaml
import shutil
import time
import logging
import pdb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from matplotlib import cm
import pysr
from pysr import PySRRegressor
from sympy import lambdify
from sympy import symbols
from sklearn.linear_model import LinearRegression


pysr.silence_julia_warning()

with open('config.yaml') as f :
    config = yaml.load(f, Loader=yaml.Loader)

if not os.path.isdir(config['outdir']):
    os.mkdir(config['outdir'])

shutil.copy('config.yaml', config['outdir']) 

logging.basicConfig(filename=os.path.join(config['outdir'],'log.log'))

# Load 2SLAQ LRG Data
df_train = pd.read_csv(config['train_filepath'], header=None, sep=' ')
df_val = pd.read_csv(config['val_filepath'], header=None, sep=' ')
df_train = pd.concat([df_train, df_val])
df_test = pd.read_csv(config['test_filepath'], header=None, sep=' ')

X_train = df_train.iloc[:,:5].to_numpy()
y_train = df_train.iloc[:,-1].to_numpy()
X_test = df_test.iloc[:,:5].to_numpy()
y_test = df_test.iloc[:,-1].to_numpy()

starttime = time.time()

model = LinearRegression().fit(X_train, y_train)

endtime = time.time()

print('Time take to fit model - ', (endtime-starttime)//60, ' minutes')
logging.info('Time take to fit model - ', (endtime-starttime)//60)

print('Score : ', model.score(X_train, y_train))
print('coeff : ', model.coef_)

y=y_test
y_predict=model.predict(X_test)
err = np.sqrt(np.sum(np.square(np.tile(model.coef_, (y_test.shape[0],1))*df_test.iloc[:,5:-1].to_numpy()), axis=1))
fig=plt.figure()
ax=fig.add_subplot(111)
H,xedges,yedges=np.histogram2d(y,y_predict, bins=50)
level=np.linspace(0,np.round(2*np.max(np.log(np.transpose(H+1))))/2.0,20)
ax.set_facecolor('black')
xe=np.zeros(len(xedges)-1)
ye=np.zeros(len(yedges)-1)
for i in range(len(xedges)-1):
    xe[i]=0.5*(xedges[i+1]+xedges[i])
    ye[i]=0.5*(yedges[i+1]+yedges[i])

plt.contourf(xe,ye,np.log(np.transpose(H+1)),levels=level,cmap='hot')
plt.plot([min(y),max(y)],[min(y),max(y)],'-',color='grey',alpha=0.9, linewidth=1.5)
plt.ylim((min(y),max(y)))

cbar=plt.colorbar()

plt.xlabel(r'$z_{spec}$',fontsize=30)
plt.tick_params(axis='both', which='major', labelsize=20)
plt.ylabel(r'$z_{phot}$',fontsize=30)
plt.ylim((min(y),max(y)))
plt.xlim((min(y),max(y)))
cbar.set_label('$log(density)$',fontsize=20)
cbar.ax.tick_params(labelsize=20)
cbar.solids.set_edgecolor("face")

print('Metric a')
logging.info('Metric a')
outliers=y_predict[abs(y-y_predict)>0.1]
print('Catastrophic Outliers: ',outliers.shape[0]*100.0/y.shape[0],'%')
logging.info('Catastrophic Outliers: ',outliers.shape[0]*100.0/y.shape[0],'%')
print('Total rms: ', np.sqrt(np.mean((y-y_predict)**2)))
logging.info('Total rms: ', np.sqrt(np.mean((y-y_predict)**2)))
print('rms w/o outliers', np.sqrt(np.mean(((y-y_predict)[abs(y-y_predict)<0.1])**2)))
logging.info('rms w/o outliers', np.sqrt(np.mean(((y-y_predict)[abs(y-y_predict)<0.1])**2)))
print('Bias:', np.mean(y-y_predict))
logging.info('Bias:', np.mean(y-y_predict))
plt.savefig(os.path.join(config['outdir'],'density_plot.png'))
plt.show()

# plot of predictions vs truth
plt.plot(y, y_predict, '.')
plt.plot([min(y),max(y)],[min(y),max(y)],'-',color='grey',alpha=0.9, linewidth=1.5)
plt.xlabel(r'$z_{spec}$',fontsize=30)
plt.tick_params(axis='both', which='major', labelsize=20)
plt.ylabel(r'$z_{phot}$',fontsize=30)
plt.ylim((min(y),max(y)))
plt.xlim((min(y),max(y)))
plt.savefig(os.path.join(config['outdir'],'predictions.png'))
plt.show()

# plot of predictions with error for 300 subsamples of galaxies

rng = np.random.default_rng(12345)
idx = rng.choice(range(len(y)), 300)
plt.errorbar(y[idx], y_predict[idx], yerr=err[idx], fmt='.', color='blue')
plt.plot([min(y),max(y)],[min(y),max(y)],'-',color='r')
plt.xlabel(r'$z_{spec}$',fontsize=30)
plt.tick_params(axis='both', which='major', labelsize=20)
plt.ylabel(r'$z_{phot}$',fontsize=30)
plt.ylim((min(y),max(y)))
plt.xlim((min(y),max(y)))
plt.savefig(os.path.join(config['outdir'],'predictions_with_error.png'))
plt.show()
