import numpy as np

def compute_bti(attr, conf, em, weight=(0.4,0.3,0.3)):
    return weight[0]*attr + weight[1]*conf + weight[2]*em

def compute_ece(confidences, labels, bins=10):
    bins = np.linspace(0,1,bins+1)
    ece = 0.0
    for i in range(len(bins)-1):
        lower, upper = bins[i], bins[i+1]
        idx = [j for j,c in enumerate(confidences) if lower<=c<upper]
        if idx:
            acc = np.mean([labels[j] for j in idx])
            conf_avg = np.mean([confidences[j] for j in idx])
            ece += abs(acc-conf_avg)*(len(idx)/len(confidences))
    return ece
