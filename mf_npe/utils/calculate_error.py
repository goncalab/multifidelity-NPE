import numpy as np
import torch
from scipy import stats
from scipy.stats import norm
import pandas as pd
from scipy.stats import t

def get_mean_ci(data, confidence=0.95):
    """
    Calculate the mean and confidence interval (95%) of the data
    like in the benchmarking paper of sbi. 
    
    State the measurement explicitely in the paper!
    """
    M = data.clone().detach() #torch.tensor(data)
    mu = torch.mean(M, dim=0)
    sigma = torch.std(M, dim=0)    
    N = M.shape[0]
    
    # divide by sqrt(N) because we are averaging over N samples
    ci = stats.norm.interval(confidence, loc=mu, scale=sigma/np.sqrt(N))
    mu = mu.cpu().numpy()
    ci_distance = mu - ci
    
    print(f"Mean: {mu}, CI: {ci_distance}")
    
    # If ci_distance is None, set it to 0
    if ci_distance is None:
        ci_distance = 0
    
    return mu, ci_distance


# This function is for the outer loop to average
def mean_confidence_interval(data, confidence=0.95):
    n = len(data)
    mean = np.mean(data)
    std_err = np.std(data, ddof=1) / np.sqrt(n)  # Standard error of the mean
    z_score = norm.ppf((1 + confidence) / 2)  # Z-score for the given confidence level
    margin_of_error = z_score * std_err
    return pd.DataFrame({
        'mean': [mean],
        'ci_min': [margin_of_error],
        'ci_max': [margin_of_error]
    })

# Input: series of values, not a matrix
def ci95_t(series):
    n = len(series)
    mean = np.mean(series)
    sem = np.std(series, ddof=1) / np.sqrt(n)
    t_crit = t.ppf(0.975, df=n - 1)
    ci = t_crit * sem
    return mean, ci