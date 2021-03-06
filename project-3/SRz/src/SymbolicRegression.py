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
import sympy

#pysr.silence_julia_warning()

with open('config.yaml') as f :
    config = yaml.load(f, Loader=yaml.Loader)

if not os.path.isdir(config['outdir']):
    os.mkdir(config['outdir'])

shutil.copy('config.yaml', config['outdir']) 

logging.basicConfig(
        filename=os.path.join(config['outdir'],'logfile.log'), 
        level=logging.DEBUG, 
        filemode='w', 
        format='%(asctime)s:%(levelname)s:%(message)s'
    )

# Load 2SLAQ LRG Data
df_train = pd.read_csv(config['train_filepath'], header=None, sep=' ')

if config['val_filepath']: # if validation data exists, merge it with train, PySR will take care of val split
    df_val = pd.read_csv(config['val_filepath'], header=None, sep=' ')
    df_train = pd.concat([df_train, df_val])

df_test = pd.read_csv(config['test_filepath'], header=None, sep=' ')

X_train = df_train.iloc[:,:5].to_numpy()
y_train = df_train.iloc[:,10].to_numpy()
X_test = df_test.iloc[:,:5].to_numpy()
X_err = df_test.iloc[:,5:10].to_numpy()
y_test = df_test.iloc[:,10].to_numpy()

#weights = 1./ y_train

model = PySRRegressor(
    populations = config['populations'],
    population_size = config['population_size'],
    niterations = config['niterations'],
    #batching = config['batching'],
    #batch_size = config['batch_size'],
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

    logger = logging.getLogger('eq'+str(k))
    file_handler = logging.FileHandler(os.path.join(config['outdir'],'eq'+str(k),'logfile.log'), mode='w')
    formatter = logging.Formatter('%(levelname)s:%(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.setLevel(logging.DEBUG)
    
    y = y_test
    y_predict = model.equations['lambda_format'][k](X_test)

    def heatmap_predictions() :
        
        H,xedges, yedges = np.histogram2d(y,y_predict, bins=50)
        level = np.linspace(0,np.round(2*np.max(np.log(np.transpose(H+1))))/2.0,20)
        xe = np.zeros(len(xedges)-1)
        ye = np.zeros(len(yedges)-1)
        for i in range(len(xedges)-1):
            xe[i] = 0.5*(xedges[i+1]+xedges[i])
            ye[i] = 0.5*(yedges[i+1]+yedges[i])
        fig, ax = plt.subplots()
        ax.set_facecolor(color='black')
        plt.contourf(xe,ye,np.log(np.transpose(H+1)),levels=level,cmap='hot')
        plt.plot([min(y),max(y)],[min(y),max(y)],'-',color='grey',alpha=0.9, linewidth=1.5)
        plt.xlabel(r'$z_{spec}$')
        plt.ylabel(r'$z_{phot}$')
        plt.ylim(min(y), max(y))
        plt.xlim(min(y), max(y))
        plt.title('Logarithmic density of true vs predicted redshifts of ~3000 galaxies in test set')

        cbar = plt.colorbar()
        cbar.set_label('$log(density)$',fontsize=20)
        cbar.ax.tick_params(labelsize=20)
        cbar.solids.set_edgecolor("face")

        plt.savefig(os.path.join(config['outdir'],'eq'+str(k),'density_plot.png'))
        #plt.show()
        plt.close()

    heatmap_predictions()

    print('Metric a')
    logger.info('Metric a')
    outliers=y_predict[abs(y-y_predict)>0.1]
    print(f'Catastrophic Outliers: {outliers.shape[0]*100.0/y.shape[0]} %')
    logger.info(f'Catastrophic Outliers: {outliers.shape[0]*100.0/y.shape[0]} %')
    print(f'Total rms: {np.sqrt(np.mean((y-y_predict)**2))}')
    logger.info(f'Total rms: {np.sqrt(np.mean((y-y_predict)**2))}')
    print(f'rms w/o outliers {np.sqrt(np.mean(((y-y_predict)[abs(y-y_predict)<0.1])**2))}')
    logger.info(f'rms w/o outliers {np.sqrt(np.mean(((y-y_predict)[abs(y-y_predict)<0.1])**2))}')
    print(f'Bias: {np.mean(y-y_predict)}')
    logger.info(f'Bias: {np.mean(y-y_predict)}')
       
    def plot_z_photo_vs_z_spec ():

        # plot of predictions vs truth
        plt.figure() 
        plt.plot(y, y_predict, '.')
        plt.plot([min(y),max(y)],[min(y),max(y)],'-',color='grey',alpha=0.9, linewidth=1.5)
        plt.xlabel(r'$z_{spec}$')
        #plt.tick_params(axis='both', which='major', labelsize=20)
        plt.ylabel(r'$z_{phot}$')
        plt.ylim((min(y),max(y)))
        plt.xlim((min(y),max(y)))
        plt.title('Photometric redshift prediction for test set')
        plt.savefig(os.path.join(config['outdir'],'eq'+str(k),'predictions.png'))
        #plt.show()
        plt.close()

    plot_z_photo_vs_z_spec()

    def calculate_errors(expr) :
        # source : https://github.com/HDembinski/essays/blob/master/error_propagation_with_sympy.ipynb

        expr = sympy.parse_expr(expr)

        symbols = sympy.symbols('u g r i z')
        err_symbols = sympy.symbols('err_u err_g err_r err_i err_z')

        expr2 = sum(expr.diff(s) ** 2 * c**2 for s, c in zip(symbols, err_symbols))
        expr2 = expr2.simplify() # recommended for speed and accuracy

        #fval = sympy.lambdify(symbols, expr)
        fcov = sympy.lambdify(symbols + err_symbols, expr2, 'numpy')

        def fn(variables, errs):
            return np.sqrt(fcov(*variables, *errs)) #fval(*x), fcov(*x, *c)

        return fn

    err = calculate_errors(model.equations['equation'][k])(np.transpose(X_test), np.transpose(X_err))

    def plot_predictions_with_errors() :

        # plot of predictions with error for 300 subsamples of galaxies
        rng = np.random.default_rng(12345)
        def stratified_sample() :
            interval_size = 0.1
            sample_size = int(300 / (max(y) / interval_size))
            l = np.arange(0, max(y), 0.1)
            idx = np.empty(0, dtype='int') 
            for i in l :
                cur_idx = ((y >= i) & (y < i+interval_size)).nonzero()[0]
                if len(cur_idx) > sample_size :
                    cur_idx = rng.choice(cur_idx, sample_size)

                idx = np.concatenate([idx, cur_idx])
            return idx

        #idx = rng.choice(range(len(y)), 300)
        idx = stratified_sample()

        plt.figure() 
        plt.errorbar(y[idx], y_predict[idx], yerr=err[idx], fmt='.', linewidth=1, color='blue', capsize=5)
        plt.plot([min(y),max(y)],[min(y),max(y)],'-',color='r')
        plt.xlabel(r'$z_{spec}$')
        #plt.tick_params(axis='both', which='major', labelsize=20)
        plt.ylabel(r'$z_{phot}$')
        plt.ylim(min(y[idx])-0.05, max(y[idx])+0.05)
        plt.xlim(min(y[idx]), max(y[idx]))
        plt.title('Photometric redshift prediction and errorbars for subsample of 300 galaxies')
        plt.savefig(os.path.join(config['outdir'],'eq'+str(k),'predictions_with_error.png'))
        #plt.show()
        #plt.draw()
        #plt.pause(0.001)
        plt.close()
    
    if len(err.shape) > 0:
        plot_predictions_with_errors()

    print(f'==========================================\n')
