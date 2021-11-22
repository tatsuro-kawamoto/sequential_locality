import sys
import numpy as np
import scipy.stats as stats
from scipy.stats import norm
from scipy.special import comb
import math
import itertools

import igraph # This is not required if you use graph-tool
import graph_tool as gt # This is not required if you use igraph

#*********************
# Basic elements
#*********************
def logsum1(N):
    f1 = 0
    for k in range(1,N):
        f1 += k*np.log(k)
    return f1
    
def logsum2(N):
    f2 = 0
    for k in range(1,N):
        f2 += k*(np.log(k))**2
    return f2

def muG(N):
    f1 = logsum1(N)
    mu_G = np.log(N) - 2*f1/(N*(N-1))
    return mu_G

def sigmaG(N): #standard deviation
    f1 = logsum1(N)
    f2 = logsum2(N)

    NC2 = N*(N-1)/2
    sigma_G_sq = f2/NC2  - (f1/NC2)**2
    sigma_G = np.sqrt(sigma_G_sq)
    return sigma_G

def get_H1(edgelist,sequence):
    H1 = 0
    normalization = len(edgelist)*(len(sequence)+1)/3
    for edge in edgelist:
        s = edge[0]
        t = edge[1]
        # When s->t, there is no t->s element (because the input is an edgelist)
        H1 += abs(sequence[s] - sequence[t])
    H1 = H1/normalization
    return H1

def get_HG(edgelist,sequence,N):
    HG = 0
    for edge in edgelist:
        s = edge[0]
        t = edge[1]
        # When s->t, there is no t->s element (because the input is an edgelist)
        HG += np.log(1 - abs(sequence[s] - sequence[t])/N)
            
    HG = -HG/(muG(N)*len(edgelist))
    return HG

def edgelist_in_envelope(edgelist, r):
    edgelist_new = []
    for edge in edgelist:
        if abs(edge[0]-edge[1]) <= r:
            edgelist_new.append(edge)
    return edgelist_new
#*********************
# END: Basic elements
#*********************

#**********************************
# Empirical z1 and zG for ER graphs
#**********************************
def get_ERrandom_H1(N, M):
    H1 = 0
    norm = M*(N+1)/3
    cnt = 0
    while True:
        i = np.random.randint(1,N+1)
        j = np.random.randint(1,N+1)
        if i != j:
            H1 += abs(i-j)
            cnt += 1
        if cnt == M:
            break         
    H1 = H1/norm
    z1 = np.sqrt(2*M*(N+1)/(N-2))*(H1-1)
    return H1, z1

def get_ERrandom_HG(N, M):
    HG = 0
    norm = -muG(N)*M
    cnt = 0
    while True:
        i = np.random.randint(1,N+1)
        j = np.random.randint(1,N+1)
        if i != j:
            HG += np.log(1 -  abs(i-j)/N)
            cnt += 1
        if cnt == M:
            break      
    HG = HG/norm
    zG = muG(N)*np.sqrt(M)*(HG-1)/sigmaG(N)
    return HG, zG
#***************************************
# END: Empirical z1 and zG for ER graphs
#***************************************


#***********************************************************
# Elements for the exact p-values w.r.t. the random sequence
#***********************************************************
def count_M3(edgelist):
    M3 = 0
    for edge_pair in itertools.combinations(edgelist, 2):
        vertices = [edge_pair[0][0], edge_pair[0][1], edge_pair[1][0], edge_pair[1][1]]
        if len(vertices) == len(set(vertices))+1:
            M3 += 1
    return M3

def VarRandomH1(N,M,M3):
    return (N+1)*((5*N-8)/(5*(N+1)) + M3*(N-4)/(5*M*(N+1)) - 2*M/(5*(N+1)))/(N-2)

def VarRandomHG(N,M,M3): 
    krange = np.arange(1,N)
    beta = muG(N)*M
    EHG2 = (1/(beta**2)) * (2*beta*M*np.log(N)- (M*np.log(N))**2 + M*(U2(N)-U4(N)) + 2*M3*(U3(N)-U4(N)) + (M**2)*U4(N) )
    return M * (muG(N)/sigmaG(N))**2 * (EHG2-1)

#@jit
def U2(N):
    U2 = 0
    for k in np.arange(1,N):
        U2 += (np.log(N-k)**2) * 2*(N-k)/(N*(N-1))
    return U2

#@jit
def U3(N):
    U31 = 0
    for i in np.arange(1,N-1):
        for j in np.arange(i+1,N):
            for k in np.arange(j+1,N+1):
                U31 += 2 * np.log(N-np.abs(i-j)) * np.log(N-np.abs(j-k))
    U32 = 0
    for i in np.arange(1,N-1):
        for k in np.arange(i+1,N):
            for j in np.arange(k+1,N+1):
                U32 += 2 * np.log(N-np.abs(i-j)) * np.log(N-np.abs(j-k))
    U33 = 0
    for j in np.arange(1,N-1):
        for i in np.arange(j+1,N):
            for k in np.arange(i+1,N+1):
                U33 += 2 * np.log(N-np.abs(i-j)) * np.log(N-np.abs(j-k))

    return 1/(math.factorial(3)*comb(N,3)) * (U31 + U32 + U33)

#@jit
def U4(N):
    U41 = 0
    for i in np.arange(1,N-2):
        for j in np.arange(i+1,N-1):
            for k in np.arange(j+1,N):
                for l in np.arange(k+1,N+1):
                    U41 += 4 * np.log(N-np.abs(i-j)) * np.log(N-np.abs(k-l))
    U42 = 0
    for i in np.arange(1,N-2):
        for k in np.arange(i+1,N-1):
            for j in np.arange(k+1,N):
                for l in np.arange(j+1,N+1):
                    U42 += 4 * np.log(N-np.abs(i-j)) * np.log(N-np.abs(k-l))
                    
    U43 = 0
    for k in np.arange(1,N-2):
        for i in np.arange(k+1,N-1):
            for j in np.arange(i+1,N):
                for l in np.arange(j+1,N+1):
                    U43 += 4 * np.log(N-np.abs(i-j)) * np.log(N-np.abs(k-l))
    U44 = 0
    for i in np.arange(1,N-2):
        for k in np.arange(i+1,N-1):
            for l in np.arange(k+1,N):
                for j in np.arange(l+1,N+1):
                    U44 += 4 * np.log(N-np.abs(i-j)) * np.log(N-np.abs(k-l))
    U45 = 0
    for k in np.arange(1,N-2):
        for i in np.arange(k+1,N-1):
            for l in np.arange(i+1,N):
                for j in np.arange(l+1,N+1):
                    U45 += 4 * np.log(N-np.abs(i-j)) * np.log(N-np.abs(k-l))
    U46 = 0
    for k in np.arange(1,N-2):
        for l in np.arange(k+1,N-1):
            for i in np.arange(l+1,N):
                for j in np.arange(i+1,N+1):
                    U46 += 4 * np.log(N-np.abs(i-j)) * np.log(N-np.abs(k-l))
    
    return 1/(math.factorial(4)*comb(N,4)) * (U41 + U42 + U43 + U44 + U45 + U46)
#***********************************************************
# END: Elements for the exact p-values w.r.t. the random sequence
#***********************************************************

#***********************************************************
# Elements for the empirical p-values w.r.t. the random sequence
#***********************************************************
def get_random_z1(edgelist, N):
    sequence_random = np.arange(N)
    np.random.shuffle(sequence_random) #randomize
    H1 = get_H1(edgelist=edgelist,sequence=sequence_random)
    M = len(edgelist)
    z1 = np.sqrt(2*M*(N+1)/(N-2))*(H1-1)
    return z1

def get_random_zG(edgelist, N):
    sequence_random = np.arange(N)
    np.random.shuffle(sequence_random) #randomize
    HG = get_HG(edgelist=edgelist,sequence=sequence_random,N=N)
    zG = muG(N)*np.sqrt(len(edgelist))*(HG-1)/sigmaG(N) # standardized H_G
    return zG

#***********************************************************
# END: Elements for the empirical p-values w.r.t. the random sequence
#***********************************************************




#///////////////////////////////////////////////////////////////////////
#
# Ordered Random Graph Model (ORGM)
# 
#///////////////////////////////////////////////////////////////////////
def get_adjacency(N, edgelist):
    Adj = np.zeros((N,N))
    for e in edgelist:
        Adj[e[0], e[1]] = 1
        Adj[e[1], e[0]] = 1
    return Adj

def Adjacency_optimized(Adj, inferred_sequence):
    N = Adj.shape[0]
    X_opt_row = np.zeros((N, N))
    X_opt = np.zeros((N, N))
    seq_sort = np.argsort(inferred_sequence)
    for i in range(N):
        X_opt_row[i,:] = Adj[seq_sort[i], :]
    for j in range(N):
        X_opt[:,j] = X_opt_row[:,seq_sort[j]]

    return X_opt

def M_within_band(Adj, r):
    N = Adj.shape[0]
    M_in = 0
    for i in range(N):
        M_in += sum(Adj[i,i:r+i+1])
    return M_in

# --------------------------------------------------
# Moments of the ordered random graph model
# --------------------------------------------------
def Mean_H1_ORGM(N, M, r, M_in=None, epsilon=None, in_envelope=True, simple=True):
    """
    Mean of H_1 in the ensemble of the ordered random graphs
    r = bandwidth
    """
    Omega_in = 0.5*r*(2*N-r-1)
    Omega_out = N*(N-1)/2 - Omega_in
    if epsilon is not None:
        M_in = int(round(M/(1+epsilon*Omega_out/Omega_in)))
    elif M_in is None and epsilon is None:
        print("Either M_in or epsilon is required as an input.", file=sys.stderr)
    beta1 = M_in*(N+1)/3 if in_envelope == True else M*(N+1)/3
    M_out = M - M_in
    if simple == True:
        if M_in > Omega_in:
            print("Error: Bandwidth r is too small", file=sys.stderr)
            sys.exit()
        elif M_out > Omega_out:
            print("Error: Bandwidth r is too large", file=sys.stderr)
            sys.exit()
        
    mean_H1_in = r*(r+1)*(3*N-2*r-1)/6
    mean_H1_in *= M_in/(Omega_in*beta1)
    mean_H1_out = (N**3-N*(3*r**2+3*r+1)+r*(2*r**2+3*r+1))/6
    mean_H1_out *= M_out/(Omega_out*beta1)
    mean_H1 = mean_H1_in if in_envelope == True else mean_H1_in + mean_H1_out
    return mean_H1

def Var_H1_ORGM(N, M, r, M_in=None, epsilon=None, in_envelope=True, simple=True):
    """
    Variance of H_1 in the ensemble of the ordered random graphs
    """
    Omega_in = 0.5*r*(2*N-r-1)
    Omega_out = N*(N-1)/2 - Omega_in
    if epsilon is not None:
        M_in = int(round(M/(1+epsilon*Omega_out/Omega_in)))
    elif M_in is None and epsilon is None:
        print("Either M_in or epsilon is required as an input.")
    M_out = M - M_in
    beta1 = M_in*(N+1)/3 if in_envelope == True else M*(N+1)/3
    if simple == True:
        if M_in > Omega_in: # added in v07
            print("Error: Bandwidth r is too small", file=sys.stderr)
            sys.exit()
        elif M_out > Omega_out:
            print("Error: Bandwidth r is too large", file=sys.stderr)
            sys.exit()
    elif Omega_in <= 1: # added in v08
        print("Error: |Omega_in| has to be larger than 1", file=sys.stderr)
        sys.exit()
    elif Omega_out <= 1:
        print("Error: |Omega_out| has to be larger than 1", file=sys.stderr)
        sys.exit()
        
    H1_sq1 = (r**2*(r+1)**2)/6 * (N*(2*r+1)/(r*(r+1))-3/2)
    H1_sq2 = (N-r)*(N-r-1)/12 * ((N+r+1/2)**2 + 2*r*(r+1)-1/4)
    H1_sq3 = (r**2*(r+1)**2)/6*(((3*N-2*r-1)**2)/6 - N*(2*r+1)/(r*(r+1)) + 3/2)
    H1_sq4 = (N-r)*(N+2*r)*(N-r-1)*(N-r+1)*(N-r-2)*(N+2*r+2)/36
    H1_sq5 = r*(r+1)*(N-r)*(N-r-1)*(N+2*r+1)*(3*N-2*r-1)/36
    H1_sq5 *= 2*M_in*M_out/(Omega_in*Omega_out)
    if simple == True:
        H1_sq1 *= M_in/Omega_in
        H1_sq2 *= M_out/Omega_out
        H1_sq3 *= M_in*(M_in-1)/(Omega_in*(Omega_in-1))
        H1_sq4 *= M_out*(M_out-1)/(Omega_out*(Omega_out-1))
    else:
        H1_sq1 *= (M_in/Omega_in)*(Omega_in+2*M_in-1)/(Omega_in+1)
        H1_sq2 *= (M_out/Omega_out)*(Omega_out+2*M_out-1)/(Omega_out+1)
        H1_sq3 *= M_in*(M_in-1)/(Omega_in*(Omega_in+1))
        H1_sq4 *= M_out*(M_out-1)/(Omega_out*(Omega_out+1))
    
    if in_envelope == True:
        H1_sq = (H1_sq1 + H1_sq3)/(beta1**2)
        if epsilon is not None:
            var_H1 = H1_sq - Mean_H1_ORGM(N,M,r,epsilon=epsilon,in_envelope=True, simple=simple)**2
        else:
            var_H1 = H1_sq - Mean_H1_ORGM(N,M,r,M_in=M_in,in_envelope=True, simple=simple)**2
    else:
        H1_sq = (H1_sq1 + H1_sq2 + H1_sq3 + H1_sq4 + H1_sq5)/(beta1**2)
        if epsilon is not None:
            var_H1 = H1_sq - Mean_H1_ORGM(N,M,r,epsilon=epsilon,in_envelope=False, simple=simple)**2
        else:
            var_H1 = H1_sq - Mean_H1_ORGM(N,M,r,M_in=M_in,in_envelope=False, simple=simple)**2

    return var_H1

# --------------------------------------------------
# END: Moments of the ordered random graph
# --------------------------------------------------


# --------------------------------------------------
# MLE bandwidth estimate in the ORGM (simple graph)
# --------------------------------------------------
def MLE_bandwidth(Adj_opt, simple=True):
    def log_approx(n):
        # stirling_approx
        return n*np.log(n) - n if n > 0 else 0

    def approx_entropy(Omega_in, Omega_out, M_in, M_out, simple):
        """
        This entropy is equal to the negative-loglikelihood
        """
        if simple == True:
            entropy_in = log_approx(Omega_in) - log_approx(M_in) - log_approx(Omega_in - M_in)
            entropy_out = log_approx(Omega_out) - log_approx(M_out) - log_approx(Omega_out - M_out)
        else:
            Q_in = Omega_in + M_in - 1
            Q_out = Omega_out + M_out - 1
            entropy_in = log_approx(Q_in) - log_approx(M_in) - log_approx(Omega_in - 1)
            entropy_out = log_approx(Q_out) - log_approx(M_out) - log_approx(Omega_out - 1)
        return entropy_in + entropy_out

    bandwidth_opt = 0
    N = Adj_opt.shape[0]
    M = int(np.sum(Adj_opt)/2)
    entropy_dict = {}
    for bandwidth in np.arange(1,N):
        Omega_in = 0.5*bandwidth*(2*N-bandwidth-1)
        Omega_out = N*(N-1)/2 - Omega_in
        M_in = M_within_band(Adj_opt, bandwidth)
        M_out = M - M_in
        if M_in < M: # We exclude M_in == M
            entropy = approx_entropy(Omega_in, Omega_out, M_in, M_out, simple)
            entropy_dict[bandwidth] = entropy
    bandwidth_opt = min(entropy_dict, key=entropy_dict.get)

    return bandwidth_opt
# --------------------------------------------------
# END: MLE bandwidth estimate in the ORGM
# --------------------------------------------------
#///////////////////////////////////////////////////////////////////////
#
# END: Ordered Random Graph Model (ORGM)
# 
#///////////////////////////////////////////////////////////////////////



class SequentialLocality:
    def __init__(self, g=None, edgelist=None, sequence=None, format='igraph', simple=True):
        if g is not None:
            if format == 'igraph':
                # igraph
                self.g = g.copy()
                self.edgelist = g.get_edgelist()
                self.M = len(self.edgelist)
                self.N = len(g.vs)
            else:
                # graph-tool
                self.g = g.copy()
                self.edgelist = self.g.get_edges()
                self.M = len(self.edgelist)
                self.N = self.g.num_vertices()
        elif edgelist is not None:
            self.edgelist = edgelist.copy()
            self.M = len(self.edgelist)
            self.N = max(set(itertools.chain.from_iterable(self.edgelist)))
        else:
            print("Either graph or edgelist is required as an input.", file=sys.stderr)

        self.simple = simple
        self.sequence = sequence.copy() if sequence is not None else np.arange(self.N)

    def H1(self, random_sequence='analytical', n_samples=10000, in_envelope=False, r=None):
        # r = bandwidth used when in_envelope == True
        if in_envelope == True:
            Adj = get_adjacency(self.N, self.edgelist)
            Adj_opt = Adjacency_optimized(Adj, self.sequence)
            bandwidth_opt = MLE_bandwidth(Adj_opt, simple=self.simple) if r is None else r
            self.edgelist = edgelist_in_envelope(self.edgelist, r=bandwidth_opt)
            self.M = len(self.edgelist)
        else:
            bandwidth_opt = None
        H1_ = get_H1(self.edgelist,self.sequence)
        z1 = np.sqrt(2*self.M*(self.N + 1)/(self.N-2))*(H1_-1) # standardized H1

        if in_envelope == False:
            # p-value w.r.t. the ER random graph
            pvalue_ER = norm.cdf(z1, loc=0, scale=1)
        else:
            M_in = M_within_band(Adj=Adj_opt, r=bandwidth_opt)
            H1mean_theory = Mean_H1_ORGM(self.N,self.M,r=bandwidth_opt,M_in=M_in,in_envelope=True,simple=self.simple)
            H1var_theory = Var_H1_ORGM(self.N,self.M,r=bandwidth_opt,M_in=M_in,in_envelope=True,simple=self.simple)
            pvalue_ER = norm.cdf(H1_,H1mean_theory,np.sqrt(H1var_theory)) # p-value based on the ORGM

        if random_sequence=='empirical':
            # Empirical p-value w.r.t. the random sequence
            pvalue_random = 0
            for sm in range(n_samples):
                if get_random_z1(self.edgelist, self.N) < z1:
                    pvalue_random += 1
            pvalue_random = pvalue_random/n_samples
        else:
            # Analytical p-value w.r.t. the random sequence (assuming the normal distribution)
            M3 = count_M3(self.edgelist)
            pvalue_random = norm.cdf(z1, loc=0, scale=np.sqrt(VarRandomH1(self.N,self.M,M3)))

        return {'H1': H1_, 'z1': z1, 'H1 p-value (ER/ORGM)': pvalue_ER, 'H1 p-value (random)': pvalue_random, 
                'bandwidth_opt': bandwidth_opt, 'simple_graph_assumption': self.simple}


    def HG(self, random_sequence='empirical', n_samples=10000):
        """
        H_G does not have the `in_envelope` option
        """
        HG_ = get_HG(self.edgelist,self.sequence,self.N)
        zG = muG(self.N)*np.sqrt(len(self.edgelist))*(HG_-1)/sigmaG(self.N) # standardized H_G

        # p-value w.r.t. the ER random graph
        pvalue_ER = norm.cdf(zG, loc=0, scale=1)

        if random_sequence=='empirical':
            # Empirical p-value w.r.t. the random sequence
            pvalue_random = 0
            for sm in range(n_samples):
                if get_random_zG(self.edgelist, self.N) < zG:
                    pvalue_random += 1
            pvalue_random = pvalue_random/n_samples
        else:
            # Analytical p-value w.r.t. the random sequence
            M3 = count_M3(self.edgelist)
            pvalue_random = norm.cdf(zG, loc=0, scale=np.sqrt(VarRandomHG(self.N,self.M,M3)))

        return {'HG': HG_, 'zG': zG, 'HG p-value (ER)': pvalue_ER, 'HG p-value (random)': pvalue_random}



