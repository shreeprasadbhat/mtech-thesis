def plot_photoz_vs_specz (photoz, specz, filepath=None):

    # plot of predictions vs truth
    plt.figure() 

    plt.plot(specz, photoz, '.')
    plt.plot([min(specz),max(specz)],[min(specz),max(specz)],'-',color='grey',alpha=0.9, linewidth=1.5)

    plt.xlabel(r'$z_{spec}$')
    plt.ylabel(r'$z_{phot}$')

    plt.ylim((min(specz),max(specz)))
    plt.xlim((min(specz),max(specz)))

    plt.title('Photometric redshift prediction for test set')

    if filepath :
        plt.savefig(filepath)

    #plt.show()
    plt.close()

if __name__ == '__main__':

    import numpy as np
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(12345)
    specz = np.linspace(0,1,100)
    photoz = specz + rng.normal(0,5,100)

    plot_photoz_vs_specz(photoz, specz, 'photoz_vs_specz.png')
