def plot_heatmap(y_true, y_pred, err, filepath=None) :
        
    H, xedges, yedges = np.histogram2d(y_true,y_pred, bins=50)
    
    level = np.linspace(0,np.round(2*np.max(np.log(np.transpose(H+1))))/2.0,20)
    
    xe = np.zeros(len(xedges)-1)
    ye = np.zeros(len(yedges)-1)
    for i in range(len(xedges)-1):
        xe[i] = 0.5*(xedges[i+1]+xedges[i])
        ye[i] = 0.5*(yedges[i+1]+yedges[i])
    
    fig, ax = plt.subplots()
    ax.set_facecolor(color='black')
    
    plt.contourf(xe,ye,np.log(np.transpose(H+1)),levels=level,cmap='hot')
    plt.plot([min(y_true),max(y_true)],[min(y_true),max(y_true)],'-',color='grey',alpha=0.9, linewidth=1.5)
    plt.xlabel(r'$z_{spec}$')
    plt.ylabel(r'$z_{phot}$')
    plt.ylim(min(y_true), max(y_true))
    plt.xlim(min(y_true), max(y_true))
    plt.title('Logarithmic density of true vs predicted redshifts of ~3000 galaxies in test set')

    cbar = plt.colorbar()
    cbar.set_label('$log(density)$',fontsize=20)
    cbar.ax.tick_params(labelsize=20)
    cbar.solids.set_edgecolor("face")

    if filepath :
        plt.savefig(filepath)
    #plt.show()
    plt.close()

if __name__ == '__main__':

    import numpy as np
    import matplotlib.pyplot as plt

    # test 
    rng = np.random.default_rng(12345)
    y_true = np.linspace(0,1,1000)
    y_pred = y_true + rng.normal(0, 0.1, 1000)
    err = rng.normal(0, 0.5, 1000)

    plot_heatmap(y_true, y_pred, err, 'heatmap.png')
