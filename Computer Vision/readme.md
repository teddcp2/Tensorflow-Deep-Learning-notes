# CNN (CONVNETS)

- Neural networks with Convolution
  
## Convolution

- Image modifier/ Pattern finder/ does the feature transformation on image
  
- So, Convoultion is an operation through which a new modified image generates i.e an image modifier. How it does? A filter/kernel will be there which will be applied on the image i.e convolution will be applied and a new modified image will be generated (e.g. blurred/edge-highlighted image) `In short Pattern finder which generates Feature maps`
  
- Kind of cross-corelation. Corelation tells the sameness b/w 2 varibables i.e. similarity b/w x and y. Dot product tell the similarity between 2.

> Image * Filter = modified-image (blurred/edge detetected etc) i.e highlighting the features for NN to learn
> 2 modes are there - valid ,same, full. (PADDING) 
> Valid will stop on the edge/boundaries as a result o/p shape will decrease where same will produce the same output by padding 0 aroung the i/p
> Full will add further padding around the i/p to cover each pixel and output will be N+k-1

- Summary of modes `Input length : N i.e. N*N,  Filter size : K i.e K*K`

> K stays ODD (3,5,7 etc)

| Mode  | Output size                                  |
| ----- | -------------------------------------------- |
| valid | N-K+1  i.e NO PADDING                        |
| same  | N    i.e N+2P-k+1 where input length is N+2P |
| Full  | N+k-1                                        |


`p is padding amount...` To make the o/p as same as i/p shape, we will need `p= f-1/2` paddings.. as `n+2p-f+1=n`


```
For colored Images
----------------------

One Filter 
    I/p    : H * W * 3
    Kernel : 3* K * K
    O/P    : (H-K+1) * (W-K+1)  Summing all the dimensions... 3rd dimension disappears

Multiple Filters 
    I/p             : H * W * 3
    Kernels(K1, K2) :  3* K * K,  3* K * K
    O/P             : (h1*w1),  (h2*w2)  on stacking both, it will be (h*w*2)

Generalization
    B = A * w 

    shape(A) > h * w * c1 
    shape(w) > c1 * K * K * c2 
    shape(B) > H * W * c2 

    Final dimension represents color channel initially and later tells the number of feature maps

Shape of Bias term
------------------------
 1. In Dense layer, W.T * X is a vector of size M , so does bias B. But in CNN, size is of c2 (inner most dimension) as W.T * X has shape of H * W * C2

 padding
 --------------
 1. Shrinking Output , for deep conv nets, in lower layers, lesser size image will be provided
 2. throwing away info from corners
 3. To solve these, we can use padding where `valid will drop the corners and reduce the o/p size and same will keep the size equal by evaluation k-1/2`

 Strides
 ---------------------
 1. After doing onetime Convoultion , it will decide the step-size and proceed to select the next input for Convolution
 2. strides will be floor{[ (n+2p-f) / S] + 1 }

```

## Pooling 

- Down-sampling i.e output a smaller image from a bigger image i.e. reducing dimension (For i/p of 100*100 with pool size 2, it will be 50* 50)
- 2 types : `max and avg pooling`. 
- Shrinking the image => lesser data to process 
- Translational Invariance i.e dont care about the feautre position, it just good that it got detected (be it on top/bottom/left/right)
- Strides are there which tells the matrix to leave the number of cells for next calculation... pool size of 2 with stride 2 is common. If u take stride 1, boxes will overlap
- why pooling after convoltion ? Even though image will shrink after each conv+pool, filter size will stay same and i/p image size will decrease, thus filter is finding more on more patterns on a particular part 
- Instead of pooling, we can use the different stride of conv layers which will reduce the size of image...  
- normally we increase number of feature maps as we go down to accomodate the decrement in image size

## Dense layer (Fully connected n/w)

- expects a 1D vector.. we can use the `Flatten or GlobalMaxPooling`. Flatten can cause problem if we have different input size.. plus it will hve more params to train. Where Global-Max will have the shape of 1 * 1 * C i.e. vector of C regardless of H and W i.e Takes max over each feature map..
  




## TODO

- Write the convolution Pseudo code, library functions
- convolution equation (1D, 2D, 3D, colored images)
- Translational Invariance , sharing the weights (ANN vs CNN), Parameter sharing
- Batch Normalization
- Cross-corelation vs convolution
- 
