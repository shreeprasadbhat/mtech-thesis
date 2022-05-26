__author__ = 'Robert Hogan'
'''
Script to plot predictions of model vs. true value for test set
'''


import numpy as np
import matplotlib.pyplot as plt
data=np.loadtxt('prediction_out')
y=data[:,0]
y_predict=data[:,1]
err=data[:,2]
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
outliers=y_predict[abs(y-y_predict)>0.1]
print('Catastrophic Outliers: ',outliers.shape[0]*100.0/y.shape[0],'%')
print('Total rms: ', np.sqrt(np.mean((y-y_predict)**2)))
print('rms w/o outliers', np.sqrt(np.mean(((y-y_predict)[abs(y-y_predict)<0.1])**2)))
print('Bias:', np.mean(y-y_predict))
plt.savefig('density_plot.png')
plt.show()

# plot of predictions vs truth
plt.plot(y, y_predict, '.')
plt.plot([min(y),max(y)],[min(y),max(y)],'-',color='grey',alpha=0.9, linewidth=1.5)
plt.xlabel(r'$z_{spec}$',fontsize=30)
plt.tick_params(axis='both', which='major', labelsize=20)
plt.ylabel(r'$z_{phot}$',fontsize=30)
plt.ylim((min(y),max(y)))
plt.xlim((min(y),max(y)))
plt.savefig('predictions.png')
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
plt.savefig('predictions_with_error.png')
plt.show()
