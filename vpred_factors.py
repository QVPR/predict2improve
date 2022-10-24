import numpy as np
from vpred_tools import find_nth_best_match_distances

def find_va_factor(S):
    d1=find_nth_best_match_distances(S,2)-find_nth_best_match_distances(S,1)
    d2=S.max(axis=0)-find_nth_best_match_distances(S,1)
    return d1/d2

def find_grad_factor(S):
    qry_list=np.arange(S.shape[1])
    g=np.zeros(len(qry_list))
    for q in qry_list:
        Sv=S[:,q]
        m0=Sv.min()
        m0_index=Sv.argmin()
        if m0_index == 0:
            g[q] = Sv[1]-Sv[0]
        elif m0_index == len(Sv)-1:
            g[q] = Sv[-2]-Sv[-1]
        else:
            g1=Sv[m0_index-1]-m0
            g2=Sv[m0_index+1]-m0
            g[q]=g1+g2
    return g