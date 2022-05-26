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

if __name__ == '__main__':

