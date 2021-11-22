import random
import numpy as np

def edge_candidates(N, bandwidth=None):
    if bandwidth is None:
        bandwidth = N-1
    edge_candidates_in = []
    edge_candidates_out = []
    for i in range(N):
        v_range = i + bandwidth
        if v_range >= N-1:
            v_range = N-1
        else:
            for k in range(v_range+1,N):
                edge_candidates_out.append((i,k))

        for j in range(i+1,v_range+1):
            edge_candidates_in.append((i,j))
            
    return edge_candidates_in, edge_candidates_out

##################################################
"""
Functions for the multigraph instances
"""
def Adj_index_weight(n_stars_bars, bars):
    # Number of edges for each adjacency matrix element in an array format
    bar_prev = -1
    Adj_ind = []
    if n_stars_bars > 0:
        for bar in bars:
            Adj_ind.append(bar - bar_prev - 1)
            bar_prev = bar
        Adj_ind.append(n_stars_bars - bar_prev - 1)
    return Adj_ind

def stars_and_bars(n_stars_bars, Omega):
    if n_stars_bars > 0:
        bars = random.sample(range(n_stars_bars), Omega-1)
        bars.sort()
    else:
        bars = []
    return bars
    
def get_edgelist(edge_candidates, Adj_ind):
    edgelist = []
    if len(Adj_ind) > 0:
        for i in range(len(Adj_ind)):
            n_edges = Adj_ind[i]
            for k in range(n_edges):
                edgelist.append(edge_candidates[i])
    return edgelist
##################################################


def ORGM(N, M, bandwidth, epsilon, simple=True):
    valid = True
    edge_candidates_in, edge_candidates_out = edge_candidates(N, bandwidth)
    Omega_in = len(edge_candidates_in)
    Omega_out = len(edge_candidates_out)
    M_in = int(round(M/(1+epsilon*Omega_out/Omega_in)))
    M_out = M - M_in
    
    if simple == True:
        if Omega_in <= M_in:
            valid = False
            edgelist = None
        else:
            np.random.shuffle(edge_candidates_in)
            np.random.shuffle(edge_candidates_out)
            edgelist = edge_candidates_in[0:M_in]
            edgelist.extend(edge_candidates_out[0:M_out])
    else:
        n_stars_bars_in = Omega_in + M_in -1
        n_stars_bars_out = Omega_out + M_out -1
        bars_in = stars_and_bars(n_stars_bars_in, Omega_in)
        bars_out = stars_and_bars(n_stars_bars_out, Omega_out)
        Adj_ind_in = Adj_index_weight(n_stars_bars_in, bars_in)
        Adj_ind_out = Adj_index_weight(n_stars_bars_out, bars_out)

        edgelist = get_edgelist(edge_candidates_in, Adj_ind_in)
        edgelist.extend(get_edgelist(edge_candidates_out, Adj_ind_out))

    return edgelist, valid
