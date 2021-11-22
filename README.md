# Statistical tests for the sequential locality of graphs

You can assess the statistical significance of the sequential locality of an adjacency matrix (graph + vertex sequence) using `sequential_locality.py`. 

This file also includes `ORGM.py` that generates an instance of the ordered random graph model (ORGM) [1] and `spectral.py` that yields an optimized vertex sequence based on the spectral ordering algorithms. 

Please find Ref. [1] for the details of the statistical tests.

# sequential_locality.py
`sequential_locality.py` executes statistical tests with respect to the sequential locality.

### Simple example
```
import numpy as np
import igraph
import sequential_locality as seq

s = seq.SequentialLocality(
		g = igraph.Graph.Erdos_Renyi(n=20,m=80), 
		sequence = np.arange(20)
		)
s.H1()
```

```
{'H1': 1.0375,
 'z1': 0.5123475382979811,
 'H1 p-value (ER/ORGM)': 0.6957960998835012,
 'H1 p-value (random)': 0.7438939644617626,
 'bandwidth_opt': None}
```

Please find `Demo.ipynb` for more examples.


## SequentialLocality
This is a class to be instantiated to assess the sequential locality.

### Input parameters
Either `g` or `edgelist` must be provided as an input.

| Parameter | Value                        | Default    | Description                                                                                                                                                                                                                                                  | 
| --------- | ---------------------------- | ---------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | 
| g         | graph                        | None       | Graph (undirected, unweighted, no self-loops) in igraph or graph-tool.                                                                                                                                                                                       | 
| edgelist  | list of tuples               | None       | Edgelist as a list of tuples.                                                                                                                                                                                                                                | 
| sequence  | 1-dim array                  | None       | Array (list or ndarray) indicating the vertex ordering. If provided, the vertex indices in the graph will be replaced based on `sequence `. If `sequence ` is `None`, the intrinsic vertex indices in the graph or edgelist will be used as the `sequence `. | 
| format    | `'igraph'` or `'graph-tool'` | `'igraph'` | Input graph format                                                                                                                                                                                                                                           | 
| simple    | Boolean                      | True       | If `True`, the graph is assumed to be a simple graph, otherwise the graph is assumed to be a multigraph.                                                                                                                                                     |                                                                                                                                                                                | 


## H1
This is a method that returns H<sub>1</sub> and z<sub>1</sub> test statistics and p-values of the input data.

### Input parameters

| Parameter       | Value                           | Default        | Description                                                                                                                                                                                                                                                        | 
| --------------- | ------------------------------- | -------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | 
| random_sequence | `'analytical'` or `'empirical'` | `'analytical'` | If `'analytical'` is selected, the p-value based on the normal approximation will be returned for the test of vertex sequence `H1 p-value (random)`. If `'empirical'` is selected, the p-value based on random sequences specified by `samples` will be returned.  | 
| n_samples       | Integer                         | 10,000         | Number of samples to be drawn as a set of random sequences. This is used only when `random_sequence = 'empirical'`.                                                                                                                                                | 
| in_envelope     | Boolean                         | `False`          | If `False`, the p-value based on the ER model will be returned. If `True`, the p-value based on the ORGM will be returned. That is, the matrix elements outside of the bandwidth `r` will be ignored.                                                              | 
| r               | Integer                         | None           | An integer between `1` and `N-1`. If provided, `r` will be used as the bandwidth when `in_envelope=True`.                                                                                                                                                          | 



### Output parameters

| Parameter            | Description                                                                                                                | 
| -------------------- | -------------------------------------------------------------------------------------------------------------------------- | 
| H1                   | H<sub>1</sub> test statistic of the input data (graph & vertex sequence)                                                           | 
| z1                   | z<sub>1</sub> test statistic of the input data                                                                                     | 
| H1 p-value (ER/ORGM) | p-value under the null hypothesis of the ER random graph (when `in_envelope=False`) or the ORGM (when `in_envelope=True`). | 
| H1 p-value (random)  | p-value under the null hypothesis of random sequences                                                                      | 
| bandwidth_opt        | Maximum likelihood estimate (MLE) of the bandwidth (when `r=None` in the input) or the input bandwidth `r`                 | 


## HG
This is a method that returns H<sub>G</sub> and z<sub>G</sub> test statistics and p-values of the input data.

- There is no `in_envelope` option for the test based on H<sub>G</sub>.
- `random_sequence = 'analytical'` can be computationally demanding.

### Input parameters

| Parameter       | Value                           | Default        | Description                                                                                                                                                                                                                                                        | 
| --------------- | ------------------------------- | -------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | 
| random_sequence | `'analytical'` or `'empirical'` | `'empirical'` | If `'analytical'` is selected, the p-value based on the normal approximation will be returned for the test of vertex sequence `H1 p-value (random)`. If `'empirical'` is selected, the p-value based on random sequences specified by `samples` will be returned.  | 
| n_samples       | Integer                         | 10,000         | Number of samples to be drawn as a set of random sequences. This is used only when `random_sequence = 'empirical'`.                                                                                                                                                | 

### Output parameters

| Parameter            | Description                                                                                                                | 
| -------------------- | -------------------------------------------------------------------------------------------------------------------------- | 
| HG                   | H<sub>G</sub> test statistic of the input data (graph & vertex sequence)                                                           | 
| zG                   | z<sub>G</sub> test statistic of the input data                                                                                     | 
| HG p-value (ER) | p-value under the null hypothesis of the ER random graph. | 
| HG p-value (random)  | p-value under the null hypothesis of random sequences                                                                      | 



# ORGM.py
`ORGM.py` is a random graph generator. 
It generates an ORGM instance that has a desired strength of sequentially lcoal structure. 

### Simple example
```
import ORGM as orgm

edgelist, valid = orgm.ORGM(
	N=20, M=80, bandwidth=10, epsilon=0.25
	)
```

### Input parameters

| Parameter | Value            | Default        | Description                                                                                                                                                                                                                     | 
| --------- | ---------------- | -------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | 
| N         | Integer          | required input | Number of vertices                                                                                                                                                                                                              | 
| M         | Integer          | required input | Number of edges                                                                                                                                                                                                                 | 
| bandwidth | Integer          | required input | Bandwidth of the ORGM                                                                                                                                                                                                           | 
| epsilon   | Float (in [0,1]) | required input | Density ratio between the adjacency matrix elements inside & outside of the envelope. When `epsilon=1`, the ORGM becomes a uniform model. When `epsilon=0`, the nonzero matrix elements are strictly confined in the envelope.  | 
| simple    | Boolean          | `True`         | If `True`, the graph is constrained to be simple. If `False`, the graph is allowed to have multiedges.                                                                                                                          | 


# spectral.py
`spectral.py` is an implementation of the spectral ordering. 

### Simple example
```
import graph_tool.all as gt
import spectral

g_real = gt.collection.ns['karate/77']
inferred_sequence = spectral.spectral_sequence(
	g= g_real, 
	format='graph-tool'
	)
```


| Parameter  | Value                        | Default        | Description                                                                       | 
| ---------- | ---------------------------- | -------------- | --------------------------------------------------------------------------------- | 
| g          | graph                        | required input | graph (undirected, unweighted, no self-loops) in igraph or graph-tool                                                   | 
| normalized | Boolean                      | `True`           | Normalized Laplacian (`True`) vs unnormalized (combinatorial) Laplacian (`False`) | 
| format     | `'igraph'` or `'graph-tool'` | `'igraph'`     | Input graph format                                                                | 



# Citation
Please use Ref. [1] for the citation of the present code.

> [1] Tatsuro Kawamoto and Teruyoshi Kobayashi, "Sequential locality of graphs and its hypothesis testing," arxiv:*** (2021).