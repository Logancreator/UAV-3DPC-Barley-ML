import numpy as np
import pandas as pd
from scipy import stats


def calculate_custom_stats(vector):

    mean = np.mean(vector) 
    std = np.std(vector, ddof=1)
    median = np.median(vector)
    nmad = 1.4826 * np.median(np.abs(vector - median))
    
    p2_5 = np.percentile(vector, 2.5) 
    q25 = np.percentile(vector, 25)
    q75 = np.percentile(vector, 75)
    p97_5 = np.percentile(vector, 97.5)
    
    iqr = q75 - q25
    
    # 百分位距 (IPR)
    ipr90 = np.percentile(vector, 95) - np.percentile(vector, 5)
    ipr99 = np.percentile(vector, 99.5) - np.percentile(vector, 0.5)
    
    # 偏度和峰度
    skewness = stats.skew(vector)
    kurtosis = stats.kurtosis(vector)
    
    # 双权重中值 (BwMv)
    def biweight_midvariance(vector):
        median = np.median(vector)
        mad = np.median(np.abs(vector - median))
        u = (vector - median) / (9 * mad)
        w = (1 - u**2)**2
        w[np.abs(u) >= 1] = 0
        bwmv = len(vector) * np.sum(w * (vector - median)**2) / (np.sum(w) * (np.sum(w) - 1))
        return np.sqrt(bwmv)
    
    bwmv = biweight_midvariance(vector)
    
    if mean == 0:
        coefficient_of_variation = 1e-10
    else:
        coefficient_of_variation = std / mean
    
    min_val = np.min(vector)
    max_val = np.max(vector)
    range_val = max_val - min_val
    entropy = stats.entropy(vector)
    geometric_mean = stats.gmean(vector)
    harmonic_mean = stats.hmean(vector)
    mad = np.median(np.abs(vector - median))
    excess_kurtosis = kurtosis - 3
    
    custom_stats = {
        'Mean': mean,
        'Std': std,
        'Median': median,
        'NMAD': nmad,
        'BwMv': bwmv,
        'P2.5%': p2_5,
        'Q25%': q25,
        'Q75%': q75,
        'P97.5%': p97_5,
        'IQR': iqr,
        'IPR90%': ipr90,
        'IPR99%': ipr99,
        'Skewness': skewness,
        'Kurtosis': kurtosis,
        'CV': coefficient_of_variation,
        'Min': min_val,
        'Max': max_val,
        'Range': range_val,
        'Entropy': entropy,
        'Geometric Mean': geometric_mean,
        'Harmonic Mean': harmonic_mean,
        'MAD': mad,
        'Excess Kurtosis': excess_kurtosis
    }
    return custom_stats