import os
import yaml
import shutil
import time
import pdb
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from matplotlib import cm
import pysr
from pysr import PySRRegressor
from sympy import lambdify
from sympy import symbols

pysr.silence_julia_warning()

with open('config.yaml') as f :
    config = yaml.load(f, Loader=yaml.Loader)

if not os.path.isdir(config['outdir']):
    os.mkdir(config['outdir'])

shutil.copy('config.yaml', config['outdir']) 

logging.basicConfig(
        filename=os.path.join(config['outdir'],'log.log'), 
        level=logging.DEBUG, 
        filemode='w', 
        format='%(asctime)s:%(levelname)s:%(message)s'
    )

# Load 2SLAQ LRG Data
df_train = pd.read_csv(config['train_filepath'], header=None, sep=' ')
df_val = pd.read_csv(config['val_filepath'], header=None, sep=' ')
df_train = pd.concat([df_train, df_val])
df_test = pd.read_csv(config['test_filepath'], header=None, sep=' ')

X_train = df_train.iloc[:,:5].to_numpy()
y_train = df_train.iloc[:,-1].to_numpy()
X_test = df_test.iloc[:,:5].to_numpy()
y_test = df_test.iloc[:,-1].to_numpy()

#weights = 1./ np.square(np.sum(df_train.iloc[:,5:-1].to_numpy(), axis=1))

model = PySRRegressor(
    niterations = config['niterations'],
    binary_operators = config['binary_operators'],
    unary_operators = config['unary_operators'],
    extra_sympy_mappings = {'pow2': lambda x : x**2, 'pow3' : lambda x : x**3, 'pow4' : lambda x : x**4, 'pow5' : lambda x : x**5},
    variable_names = config['variable_names'],
    model_selection = config['model_selection'],
    #weights = weights,
    loss = config['loss'],
    equation_file = os.path.join(config['outdir'], config['equation_file']) ,
    progress = config['progress'] 
)

starttime = time.time()

model.fit(X_train, y_train)

endtime = time.time()

print(f'Time take to fit model - {(endtime-starttime)//60} minutes')
logging.info(f'Time take to fit model - {(endtime-starttime)//60} minutes')

print(model)
logging.info(model)

model.equations.to_csv(os.path.join(config['outdir'],'dataframe.csv'), index=False)

n_equations = model.equations.shape[0]

print(f'Best equation : {model.sympy()}\n')
logging.info(f'Best equation : {model.sympy()}\n')

for k in range(n_equations):
    equation_sympy =  model.equations['sympy_format'][k]
    print(f'==========================================\n')
    print(f'Equation {str(k)} : {equation_sympy}')
    print(f'==========================================\n')
    
    # directory to store results from each equation
    if not os.path.isdir(os.path.join(config['outdir'],'eq'+str(k))):
        os.mkdir(os.path.join(config['outdir'],'eq'+str(k)))

    logging.basicConfig(
        filename=os.path.join(config['outdir'],'eq'+str(k),'log.log'), 
        level=logging.DEBUG, 
        filemode='w', 
        format='%(asctime)s:%(levelname)s:%(message)s'
    )

    y=y_test
    y_predict=model.equations['lambda_format'][k](X_test)

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
    print(f'Catastrophic Outliers: {outliers.shape[0]*100.0/y.shape[0]} %')
    logging.info(f'Catastrophic Outliers: {outliers.shape[0]*100.0/y.shape[0]} %')
    print(f'Total rms: {np.sqrt(np.mean((y-y_predict)**2))}')
    logging.info(f'Total rms: {np.sqrt(np.mean((y-y_predict)**2))}')
    print(f'rms w/o outliers {np.sqrt(np.mean(((y-y_predict)[abs(y-y_predict)<0.1])**2))}')
    logging.info(f'rms w/o outliers {np.sqrt(np.mean(((y-y_predict)[abs(y-y_predict)<0.1])**2))}')
    print(f'Bias: {np.mean(y-y_predict)}')
    logging.info(f'Bias: {np.mean(y-y_predict)}')
    plt.savefig(os.path.join(config['outdir'],'eq'+str(k),'density_plot.png'))
    #plt.show()

    # plot of predictions vs truth
    plt.plot(y, y_predict, '.')
    plt.plot([min(y),max(y)],[min(y),max(y)],'-',color='grey',alpha=0.9, linewidth=1.5)
    plt.xlabel(r'$z_{spec}$',fontsize=30)
    plt.tick_params(axis='both', which='major', labelsize=20)
    plt.ylabel(r'$z_{phot}$',fontsize=30)
    plt.ylim((min(y),max(y)))
    plt.xlim((min(y),max(y)))
    plt.savefig(os.path.join(config['outdir'],'eq'+str(k),'predictions.png'))
    #plt.show()

    # plot of predictions with error for 300 subsamples of galaxies
    
    #rng = np.random.default_rng(12345)
    #idx = rng.choice(range(len(y)), 300)
    #plt.errorbar(y[idx], y_predict[idx], yerr=err[idx], fmt='.', color='blue')
    #plt.plot([min(y),max(y)],[min(y),max(y)],'-',color='r')
    #plt.xlabel(r'$z_{spec}$',fontsize=30)
    #plt.tick_params(axis='both', which='major', labelsize=20)
    #plt.ylabel(r'$z_{phot}$',fontsize=30)
    #plt.ylim((min(y),max(y)))
    #plt.xlim((min(y),max(y)))
    #plt.savefig(os.path.join(config['outdir'],'eq'+str(k),'predictions_with_error.png'))
    #plt.show()

    print(f'==========================================\n')

