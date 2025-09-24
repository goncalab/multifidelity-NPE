from mf_npe.fsbi.utils import get_density_estim_data, save_metric
import time



def compute_metrics():
    '''
    This function computes the metrics for the third task of the project, after loading the simulations from the cluster.
    '''
    # Polynomial rules
    metrics = ["rate","cv_isi","kl_isi","spatial_Fano","temporal_Fano","auto_cov","fft","w_blow",
           "std_rate_temporal","std_rate_spatial","std_cv","w_creep","rate_i",
           "weef","weif","wief","wiif"]

    round_name = "pi3_r5to10Hz"
    h5_path = "data_synapsesbi/" + str(round_name) + ".h5"
    
    
    start = time.time()
    output = get_density_estim_data(h5_path, metrics, parallel=True, n_workers=10) #change number of workers depending on your workstation
    print(time.time()-start, "s")
    
    
    default_path = h5_path[:-3] + "_metrics.npy"

    save_metric(path=default_path,
                data=output)