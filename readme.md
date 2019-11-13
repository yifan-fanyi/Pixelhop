last update 2019.11.13

### Run the code in `Python3` in `./src` folder.

## Pixelhop Unit -> `pixelhop.py`

```
PixelHop_Unit(feature, dilate=np.array([1]), num_AC_kernels=6, pad='reflect', weight_name='tmp.pkl', getK=False, useDC=False, batch=None):
```
From [arXiv:1909.08190](https://arxiv.org/abs/1909.08190). 


*`x`* -> Input, 4-D tensor `(N,H,W,D)`, the same as `channel_last` mode in Keras.  
*`dilate`* -> (numpy array or list) Controls location of chooesn neghbour pixels. Support any window size.  
*`pad`* -> Padding, support `none`, `reflect`, `zeros` (default: `reflect`)   
*`num_AC_kernels`* -> Number of AC components to be kept.   
*`weight_name`* -> Saab kernel file location to be saved or loaded. (default: `../weight/+weight_name`)   
*`getK`* -> If use input to compute Saab kernel. (default: `True`)  
*`useDC`* -> If add DC component. (default: `False`)  
*`batch`* -> Batch size. (default: `None`)   

## Adaptive KMeans -> `ada_kmeans.py`
```
Ada_KMeans(X, Y=None, path='tmp.pkl', train=True, sep_num=2, trial=6, batch_size=10000, minS=300, maxN=50, limit=0.5, maxiter=50):
```
*`X`* -> input feature  
*`Y`* -> corresponding labels of each training sample, testing process do not need it (default: `None`)  
*`path`* -> path to weight file to loaded or saved in `../weight/` (default: `"tmp.pkl"`)    
*`train`* -> True: training stage; False: testing stage (default: `True`)  
*`sep_num`* -> number of clusters at each split (default: `2`)  
*`trial`* -> number of trial in each split (default: `6`)  
*`batch_size`* None is not use batch process, not optimized if using batch, more time, but save memory (default: `10000`) 
*`minS`* -> minimum samples in one cluster (default: `300`)  
*`maxN`* -> maximum number of clusters (default: `50`)  
*`limit`* -> when to stop a cluster, can be self-defined by modifying the `Continue_split` function (default: `0.5`, entropy less than 0.5)  
*`maxiter`* -> number of maximum iteration (default: `50`)

Regression method can be modified in `Regression_Method` function, currently using `LogisticRegression`
# Old code, do not maintain or used
## LAG Unit -> `LAG.py`

final update: 2019.11.11  
modified based on code from `yueru`
```
feature = LAG_Unit(X, Y=None, class_list=[0,1], weight_path="LAG_weight.pkl", num_clusters=[10,10], alpha=5, batch_size=None, train=True)
```

Reference: [arXiv:1810.02786](https://arxiv.org/abs/1810.02786), Saab code is modified from https://github.com/davidsonic/Interpretable_CNN. 

*`X`* -> Input data matrix, 2-D tensor `(N,D)`  
*`Y`* -> corresponding labels of each training sample, testing process do not need it (default: `None`)  
*`class_list`* -> list of object classes (default: `None`)  
*`weight_path`* -> path to weight file to loaded or saved in `../weight/` (default: `"LAG_weight.pkl"`)  
*`num_clusters`* -> output feature dimension (default: [10,10])   
*`batch_size`* -> None is not use batch process, not optimized if using batch, more time, but save memory (default: `None`)  
*`alpha`* -> A parameter to determine the relationship between the Euclidean distance and the likelihood for a sample belonging to a cluste (default: `5`)  
*`train`* -> True: training stage; False: testing stage (default: `True`)  

#### Example:
`./src/LAG_example.py`. 

## Cross Entropy -> `cross_entropy.py`
final update: 2019.11.11
```
ce = Cross_Entropy(num_class=2, num_bin=10)
H = ce.compute(x, y, class_weight=None)
```
Reference: Manimaran A, Ramanathan T, You S, et al. Visualization, Discriminability and Applications of Interpretable Saak Features[J]. 2019.  

*`num_class`* -> number of class  
*`num_bin`* -> number of bin (default; `10`)

## Triplet Loss Unit -> `triplet.py`
Single fully connected layer without activation using triplet loss to optimize based on Keras.

```
Triplet_Unit(X, new_dim=10, a=0.6, train=True, epochs=1000, saved_name='Triplet.h5', optimizer='sgd')
```
*`X`* -> input data, organized in `[anchor : positive : negative]`  
*`new_dim`* -> dimension of embedding feature (default: `10`)  
*`a`* -> alpha in triplet loss (default: `0.6`)  
*`train`* -> if train of not (default: `True`)
*`epochs`* -> epochs of training (default: `1000`)  
*`saved_name`* -> path to weight file to loaded or saved in `../weight/` (default: `'Triplet.h5'`)  
*`optimizer`* -> optimizer in Keras optimizer (default: `sgd`)

## Rectangular Receptive Field Pixelhop -> `rect_pixelhop.py`
final update: 2019.10.25
```
rect_PixelHop_Unit(feature, xdilate=[1], ydilate=[1], num_AC_kernels=6, pad='reflect', weight_name='tmp.pkl', getK=False, useDC=False):
```
As name, no batch support.
