# Models

The Models module contains a list of model that can be used for Classification, Segmentation and Generation.

## Modules

### AlexNet

For network information, see AlexNet [paper](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf).

Parameters:

<ul>
<li>intial_feature_map_size (int,required): The input tensor size.
<li>num_classes (int,required): The number of class in the data.
<li>in_channels (int,required): The number of channels in the input tensor.
<li>strides (list,default = [1,2,1,1,1]): The strides in each convolutional layer of the AlexNet.
<li>channels (list,default = [1,2,2,2,1]): The channels in each convolutional layer of the AlexNet.
<li>kernel_size (list,default = [3,5,3,3,1]): The size of kernels in each convolutional layer of the AlexNet.
<li>padding (list,default = [0,1,1,1,1]): The padding in each convolutional layer of the AlexNet.
<li>demographic (list,default = []): The list demographic factors to be used for classification.
</ul>

Usage:
```python
from niftytorch.Models.AlexNet import AlexNet
import torch
initial_feature_map = 128
num_classes = 2
in_channels = 1
demographic = ['factor1','factor2']
strides = [1,2,1,1,1]
channels = [1,2,2,2,1]
kernel_size = [3,5,3,3,1]
padding = [0,1,1,1,1]
input = torch.rand(64,in_channels,initial_feature_map,initial_feature_map,initial_feature_map)
model = AlexNet(initial_feature_map,num_classes = num_classes,in_channels = in_channels,strides = strides,channels = channels,kernel_size = kernel_size,padding = padding,demographic = demographic)
```

### VGGNet
For network information, see VGGNet [paper](https://arxiv.org/pdf/1409.1556.pdf).

Parameters for VGG_Net:
image_scale, cfgs, version, features, num_classes,init_weights
<ul>
<li>image_scale (int,requried): The input tensor size along width size.
<li>cfgs (int,required): The configuration of VGG_Net
<li>version (str,required): The version can be 'A','B','D' or 'E'.
<li>features (torch.Tensor,required): The features from convolution layers.
<li>num_classes (int,default = 2): The classes in the dataset.
<li>init_weights (bool,default = False): The if true weights are initialzed using kaiming normal or else it is initialized randomly.
<li>demographic (list,default = []): The list demographic factors to be used for classification.
</ul>

Parameters for make_layers:

<ul>
<li>cfg (list,required): The configuration for VGG Net (cfgs[version]).
<li>in_channels (int,default = 1): The number of channels in the input tensor.
<li>batchnorm (boolean,default = True): This boolean variable is used to inidicated whether the batchnorm layer is required.
</ul>


Usage:
```python
from niftytorch.Models.VGG_Net import VGG
from niftytorch.Models.VGG_Net import make_layers
import torch
image_scale = 128
demographic = ['factor1','factor2']
cfgs = {'A':[32,32,32,'M',64,64,64]}
version = 'A'
in_channels = 1
num_classes = 2
init_weights = True
input = torch.rand(64,in_channels,image_scale,image_scale,image_scale)
model = VGG(image_scale,cfgs = cfgs,version = version,num_classes = num_classes,init_weights = init_weights,demographic = demographic)
```

### ResNet

For network information, see ResNet [paper](https://arxiv.org/abs/1512.03385).

Parameters:

<ul>
<li>block (model,required): The block can be BasicBlock or BottleNeck Layers.
<li>layers (int,required): The number of layers in the block layers.
<li>stride (int,required): The stride at each layer the size of stride is same as the number of layers.
<li>in_channels (int,required): The number of channels in the input data.
<li>num_classes (int,required): The number of class in the data.
<li>zero_init_residual (bool,default = False): This variable decides whether the batchnorm layer is to be initialized.
<li>groups (int,default = 1): The number of groups to be considered in the block layer.
<li>width_per_group (int,default = 64): The number of channels in each block layer.
<li>replace_stride_with_dilation (bool,default = False): The dilation at each block layer.
<li>norm_layer (torch.nn,default = None): The type of norm layer to be used.
<li>demographic (list,default = []): The list demographic factors to be used for classification.
</ul>

Usage:
```python
from niftytorch.Models.ResNet import ResNet
import torch
import torch.nn as nn
from niftytorch.Layers.Convolutional_Layers import Bottleneck
block = Bottleneck
demographic = ['factor1','factor2']
layers = [1,2,1,1,2]
stride = [1,1,1,1,1]
num_classes = 2
zero_init_residual = True
groups = 1
replace_stride_with_dilation = [2,2,2,2,2]
norm_layer = nn.Batchnorm3d
in_channels = 1
input = torch.rand(64,in_channels,32,32,32)
model = ResNet(block = block,layers = layers,stride = stride,in_channels = 1,num_classes = num_classes,zero_init_residual = zero_init_residual,groups = groups,replace_stride_with_dilation = replace_stride_with_dilation,norm_layer = norm_layer,demographic = demographic)
```

### ShuffleNet

For network information, see ShuffleNet [paper](https://arxiv.org/abs/1707.01083).

Parameters:

<ul>
<li>stage_repeats (list,required): The number of times each stage is repeated
<li>groups (int, deafult = 3): number of groups to be used in grouped 1x1 convolutions in each ShuffleUnit.
<li>in_channels (int, deafult = 1): number of channels in the input tensor.
<li>num_classes (int, default = 2): number of classes to predict.
<li>demographic (list,default = []): The list demographic factors to be used for classification.
</ul>

Usage:
```python
from niftytorch.Models.ShuffleNet import ShuffleNet
import torch
import torch.nn as nn
stage_repeats  = [3,7,3]
groups = 5
num_classes = 2
demographic = ['factor1','factor2']
in_channels = 1
model = ShuffleNet(stage_repeats = stage_repeats,groups = groups,in_channels = 1,num_classes = num_classes,demographic = demographic)
```

### SqueezeNet

For network information, see SqueezeNet [paper](https://arxiv.org/abs/1602.07360).

Parameters:

<ul>
<li>version (str,default = '1_0'): The version of shuffleNet to be used.
<li>num_classes (str, deafult = 2): Number of classes in the dataset.
<li>in_channels (int, deafult = 1): number of channels in the input tensor.
</ul>

Usage:
```python
from niftytorch.Models.SqueezeNet import SqueezeNet
import torch
import torch.nn as nn
version  = '1_1'
num_classes = 2
in_channels = 1
model = SqueezeNet(version = version,num_classes = 2,in_channels = 1)
```

### XNOR-Net

For network information, see XNOR-Net [paper](https://arxiv.org/abs/1603.05279).

Parameters:
<ul>
<li>image_scale: The input tensor size along width size. 
<li>num_classes: Number of classes in the dataset.
<li>in_channels: number of channels in the input tensor.
<li>channels: The channels in each convolutional layer of the AlexNet.
<li>kernel_size: The size of kernels in each convolutional layer of the AlexNet.
<li>strides: The strides in each convolutional layer of the AlexNet.
<li>padding: The padding in each convolutional layer of the AlexNet.
<li>groups: The number of groups to be considered in the block layer.
<li>demographic (list,default = []): The list demographic factors to be used for classification.
</ul>

Usage:
```python
from niftytorch.Models.XNOR_NET import AlexNet
image_scale = 128
num_classes = 2
in_channels = 1
demographic = ['factor1','factor2']
channels = [8,16,24,32,32,32]
kernel_size = [11, 5, 3, 3, 3]
strides = [4, 1, 1, 1, 1]
padding = [0, 2, 1, 1, 1]
groups = [1, 1, 1, 1, 1]
model = AlexNet(image_scale = image_scale,num_classes = num_classes,in_channels = in_channels,channels = channels,kernel_size = kernel_size,strides = strides,padding = padding,groups = groups,demographic = demographic)
```

### U-Net 

Parameters:

<ul>
<li>in_channels(int,required): The number of channels in the input data.
<li>out_channels(int,required): The number of channels in the output data.
<li>init_features(int,required): The number of initial features to which the upsampling and downsampling operations are applied to.
<li>norm_layer (torch.nn,default = None): The type of norm layer to be used.
<li>kernel_size(list,default=[]): The size of kernels in each convolutional layer of the UNet.
<li>strides(list,default=[]): The strides in each convolutional layer of the UNet.
<li>padding(list,default=[]): The padding in each convolutional block of the UNet.
</ul>


in_channels, out_channels, stride, init_features, bias, kernel_size, padding, groups = 1, norm_layer = None

Usage:
```python
from niftytorch.models.Unet import UNet
import torch
in_channels = 1
out_channels = 3
stride = [1,1,1,2,2,2,1,1]
init_features = 16
bias = [True,True,True,True,False]
groups = 1
kernel_size = [3,3,3,5,5,5,5,3,3,3,3]
norm_layer = torch.nn.BatchNorm3d
kernel_size = [3,3,5,5,5,5,4,4,5,4,1]
stride = [2,2,2,2,2,2,2,2]
padding = [1,1]
input = torch.rand(64,1,32,32,32)
model = UNet(in_channels = in_channels,out_channels = out_channels,stride = stride,init_features = init_features,bias = bias,kernel_size = kernel_size,padding = padding,groups = groups,norm_layer = norm_layer)
output = model(input)
```
