import numpy as np
from scipy.sparse.linalg import eigsh

import igraph
import graph_tool as gt

#------------------------------------------------
# Spectral algorithm for the optimal sequence
#------------------------------------------------
def Lp_eigs(g, normalized, format):
    if format == 'igraph':
        Lp = g.laplacian(normalized=normalized) # igraph
    else:
        Lp = gt.spectral.laplacian(g, norm=normalized).todense() # graph-tool
    w, v = eigsh(np.array(Lp, dtype='f'), which="SA", k=2)
    if normalized == True:
        if format == 'igraph':
            degrees = g.degree() # igraph
        else:
            degrees = g.get_total_degrees(g.get_vertices()) # graph-tool
        degree_correction = 1/np.sqrt(degrees)
        v2_ = v[:,1]
        v2 = degree_correction*v2_
    else:
        v2 = v[:,1]
    return v2

def spectral_sequence(g, normalized=True, format='igraph'):
    """
    This function returns the infeerred optimal sequence as an array.
    `inferred_sequence[i]` denotes the optimized order of the vertex that is originally indexed as i (i.e., sigma_{i}).
    """
    v2 = Lp_eigs(g, normalized=normalized, format=format)
    inferred_sequence = np.argsort(np.argsort(v2))
    return inferred_sequence
#------------------------------------------------
# END: Spectral algorithm for the optimal sequence
#------------------------------------------------
