# Loss

The Loss module consists a collection of loss functions which can be used for learning model and optimizing the weights.

## Modules 

### Weighted Cross Entropy

The `weighted_cross_entropy` loss function weights probabilities for each class with weighted assigned by the user and then normalized by total weight.<br>

Parameters:
<ul>
<li>num_class (int,required): The number of classes in the problem.
<li>weight (torch.FloatTensor,default = None) : The alpha is the weight to be used as a multiplier for probabilities of each class.
</ul>

Usage:

```python
from niftytorch.Loss.Losses import weighted_cross_entropy
import torch
import numpy as np
num_classes = 32
weight = np.random.rand(num_classes)
input = torch.random.rand(64,num_classes)
output = torch.zeroes(64,num_classes)
loss = weighted_cross_entropy(num_class = num_classes,weight = weight)
print(loss(input,output))
```

### Focal Loss

The idea behind `FocalLoss` is to not only weight the probabilities according to class imbalance, but also take into consideration the difficulty of classifying the example. 

Parameters:
<ul>
<li>num_class (int,required): The number of classes in the problem.
<li>alpha (float,default = None) : The alpha is the weight to be used as a multiplier for probabilities of each class.
<li>gamma (float,default = 2.0): The gamma is the exponent to be used in the probabilities.
<li>balance_index (int,default = -1): This is the index which needs to be balanced or has the imbalanced (This is used if alpha is a float instead of ndarray).
<li>smooth (float,required): The smoothening factor is used to smoothen ground truth labels.
<li>size_average (boolean,default = True): The size_average tells whether to average the loss if true else it the sum is returned.
</ul>

Usage:

```python
from niftytorch.Loss.Losses import FocalLoss
import torch
import numpy as np
num_classes = 32
alpha = np.random.rand(num_classes)
input = torch.random.rand(64,num_classes)
output = torch.zeroes(64,num_classes)
loss = FocalLoss(num_class = num_classes,alpha = alpha,gamma = 2.0,balance_index = 1,smooth = 0.1,size_average = True)
print(loss(input,output))
```

### Focal Dice Loss

The idea behind `FocalDiceLoss` is to not only weight the probabilities according to class imbalance, but also take into consideration the difficulty of classifying the example by 

Parameters:
<ul>
<li>alpha (float,required): The alpha is the weight to be used as a multiplier for dice loss coefficient of each class.
<li>beta (float,required): The beta is the exponent to be used in the probabilities.
<li>eps (float,required): It is a really small number which decides which used along with denominator in dice loss to avoid division by zero error. 
<li>num_class (int,required): The number of classes in the problem.
</ul>

Usage:

```python
from niftytorch.Loss.Losses import FocalDiceLoss
import torch
import numpy as np
num_class = 32
alpha = torch.random.rand(num_class)
input = torch.random.rand(64,num_class)
output = torch.zeroes(64,num_class)
loss = FocalDiceLoss(alpha = alpha,gamma = 2.0,eps = 1e-8,num_class = num_class)
print(loss(input,output))
```

### Tversky Loss

The `tversky_loss` function is used to weight false positive and false negative in the loss.

Parameters:
<ul>
<li>alpha (float,required): The alpha is the weight to be used as a multiplier for dice loss coefficient of each class.
<li>beta (float,required): The beta is the exponent to be used in the probabilities.
<li>eps (float,required): It is a really small number which decides which used along with denominator in dice loss to avoid division by zero error.
</ul>

Usage:

```python
from niftytorch.Loss.Losses import tversky_loss
import torch
import numpy as np
num_class = 32
alpha = torch.random.rand(num_class)
input = torch.random.rand(64,num_class)
output = torch.zeroes(64,num_class)
loss = tversky_loss(alpha = alpha,gamma = 2.0,eps = 1e-8)
print(loss(input,output))
```

### Contrastive Loss

The `ContrastiveLoss` function is used in few shot learning paradigm. The inputs of the contrastive loss are two input tensors and target tensor. The target is 0 if they're of different class else it is 1.

Parameters:
<ul>
<li>margin (float,required): The margin is the maximum allowed to distance between the input distances.
<li>eps (float,default = 1e-9): It is a really small number which decides which used along with denominator in dice loss to avoid division by zero error. 
<li>size_average (Boolean,default = True): If true, then the loss is averaged or else the sum of the loss is returned.
</ul>

Usage:

```python
from niftytorch.Loss.Losses import ContrastiveLoss
import torch
margin = 10.0
input1 = torch.random.rand(64,1024)
input2 = torch.random.rand(64,1024)
output = torch.zeroes(64) #assuming they're from different class if you they're from same class use torch.ones(64,num_class)
loss = ContrastiveLoss(margin = margin,eps = 1e-8,size_average = True)
print(loss(input1,input2,output))
```

### Triplet Loss

The `TripletLoss` is used in few shot learning paradigm. The inputs of the triplet loss are two tensor of different classes and an anchor tensor. The distance between the anchor and positive, the distance between the anchor and negative is used assign the class for the anchor.

Parameters:
<ul>
<li>margin (float,required): The margin is the maximum allowed to distance between the input distances.
<li>size_average (Boolean,default = True): If true, then the loss is averaged or else the sum of the loss is returned.
</ul>

Usage:

```python
from niftytorch.Loss.Losses import TripletLoss
import torch
anchor =  torch.random.rand(64,1024) #embedding for anchor
positive = torch.random.rand(64,1024) #embedding for positive sample
negative = torch.random.rand(64,1024) #embedding for negative sample
loss = TripletLoss(margin = margin,size_average = True)
print(loss(anchor,positive,negative))
```
