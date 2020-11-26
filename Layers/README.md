# Layers

The Layers module contains set of convolutional neural network units, which can be used to create custom neural network architectures.

## Modules

### Conv3x3

This is a simple conv 3x3 layer.

Parameters:
<ul>
<li>in_planes (int,required): The number of channels in the input feature map.
<li>out_planes (int,requied): The number of channels in the output feature map.
<li>stride (int, default = 1): The stride used in each convolution filter.
<li>groups (int, default = 1): The number of groups to be considered while performing convolution. 
<li>dilation (int, default = 1): The field of view to be considered while performing convolution.
<li>bias (int, default = False)
</ul>

Usage:
```python
from niftytorch.Layers.Convolutional_Layers import conv3x3
import torch
in_planes = 512
input = torch.rand(32,in_planes,32,32,32)
convolution = conv3x3(in_planes = in_planes,out_planes = 32,stride = 1,groups = 1,dilation = 1,bias = False)
output = convolution(input)
```

### Conv1x1

This is a simple conv 1x1 layer.

Parameters:
<ul>
<li>in_planes (int,required): The number of channels in the input feature map.
<li>out_planes (int,requied): The number of channels in the output feature map.
<li>stride (int, default = 1): The stride used in each convolution filter.
<li>groups (int, default = 1): The number of groups to be considered while performing convolution.
<li>bias (int, default = False)
</ul>

Usage:
```python
from niftytorch.Layers.Convolutional_Layers import conv1x1
import torch
in_planes = 512
input = torch.rand(32,in_planes,32,32,32)
convolution = conv1x1(in_planes = in_planes,out_planes = 32,stride = 1,groups = 1,bias = False)
output = convolution(input)
```

### Basic Block

`BasicBlock` is a series of convolution, batchnorm, activation layers with one jump skip connection. 

The structure of this block is as follows:

input -> conv1 -> bn1 -> relu -> conv2 -> bn2 + input -> relu : output

Parameters:

<ul>
<li>inplanes (int,required): The number of channels in the input feature map. 
<li>planes (int,required): The number of channels in the output feature map.
<li>stride (int,default = 1): The number of strides to be used in the conv1
<li>downsample (int,default = None): The downsampler to be used to reduce the feature map size F.interpolate
<li>groups (int,default = 1): The number of groups to be considered in convolution
<li>dilation (int, default = 1): The field of view to be considered while performing convolution.
<li>norm_layer (int,default = None): The normalization layer to be used ex: nn.BatchNorm()
</ul>

Usage:

```python
from niftytorch.Layers.Convolutional_Layers import BasicBlock
import torch
in_planes = 512
input = torch.rand(32,in_planes,32,32,32)
convolution_block = BasicBlock(in_planes = in_planes,planes = 256,out_planes = 32,stride = 1,groups = 1,bias = False)
output = convolution_block(input)
```

### BottleNeck Block

`BottleNeck` is a series of convolution,batchnorm,activation layers with two layer jump skip connection. 

The structure of this block is as follows:

input -> conv1 -> bn1 -> relu -> conv2 -> bn2 -> relu -> conv3 -> bn3 + input -> relu : output

Parameters:

<ul>
<li>inplanes (int,required): The number of channels in the input feature map. 
<li>planes (int,required): The number of channels in the output feature map.
<li>stride (int,default = 1): The number of strides to be used in the conv1
<li>downsample (int,default = None): The downsampler to be used to reduce the feature map size F.interpolate
<li>groups (int,default = 1): The number of groups to be considered in convolution.
<li>base_width (int,default = 1): The base width and expansion are used to calculate the number of filters for conv2 in bottleneck block.
<li>dilation (int, default = 1): The field of view to be considered while performing convolution.
<li>norm_layer (int,default = None): The normalization layer to be used ex: nn.BatchNorm()
<li>expansion (int,default = 1):The base width and expansion are used to calculate the number of filters for conv2 in bottleneck block.
</ul>

Usage:

```python
from niftytorch.Layers.Convolutional_Layers import Bottleneck
import torch
in_planes = 512
planes = 256
input = torch.rand(32,in_planes,32,32,32)
convolution_block = BottleNeck(inplanes = in_planes, planes = planes, stride=1, downsample=None, groups=1,base_width=64, dilation=1, norm_layer=None,expansion = 1)
output = convolution_block(input)
```

### ShuffleUnit


`ShuffleUnit` is the class definition for ShuffleUnit Module, which uses channel shuffling and combination to improve the network performance with fewer parameters.<br>
The ShuffleUnit is commonly used in ShuffleNet 1.0 and ShuffleNet 2.0, but they can be used along with any network. For more information, please read the ShuffleNet papers [[ShuffleNet 1.0](https://arxiv.org/abs/1707.01083), [ShuffleNet 2.0](https://arxiv.org/abs/1807.11164)].<br>

Parameters:

<ul>
<li>in_channels (int,required): number of  channels in the input tensor the shuffleunit. 
<li>out_channels (int,required): number of channels in the tensor generated from the shuffleunit.
<li>groups (int,default = 3): define number of groups in which the filters are combined before shuffle operation.
<li>grouped_conv (bool,default = True): defines whether grouped_convolution is to be used or not.
<li>combine ('add' or 'concat',default = 'add'): defines the method in which we can combine the channels it has two options add or concat.
<li>compresstion_ratio (int,default = 4): The number of channels to be required in the bottleneck channels.
</ul>

Usage:

```python
from niftytorch.Layers.Convolutional_Layers import ShuffleUnit
import torch
in_channels = 512
out_channels = 256
input = torch.rand(32,in_planes,32,32,32)
convolution_block = ShuffleUnit(in_channels = in_channels, out_channels = out_channels, groups=5, grouped_conv=True, combine = 'concat',compression_ratio = 4)
output = convolution_block(input)
```

### Fire Module


`Fire` is the class definition for Fire Module, which uses a combination of 1x1 and 3x3 filters to improve the network performance with fewer parameters.<br>
The Fire Module is commonly used in Squeezenet 1.0 and SqueezeNet 2.0 but they can used along with any network. For more information please read the [SqueezeNet paper](https://arxiv.org/abs/1602.07360).<br>

Parameters:

<ul>
<li>in_channels (int,required): number of  channels in the input tensor the shuffleunit. 
<li>out_channels (int,required): number of channels in the tensor generated from the shuffleunit.
<li>groups (int,default = 3): define number of groups in which the filters are combined before shuffle operation.
<li>grouped_conv (bool,default = True): defines whether grouped_convolution is to be used or not.
<li>combine ('add' or 'concat',default = 'add'): defines the method in which we can combine the channels it has two options add or concat.
<li>compresstion_ratio (int,default = 4): The number of channels to be required in the bottleneck channels.
</ul>

Usage:

```python
from niftytorch.Layers.Convolutional_Layers import Fire
import torch
in_planes = 20
input = torch.rand(32,20,32,32,32)
convolution_block = Fire(inplanes = in_planes, squeeze_planes = 3, expand1x1_planes = 12, expand3x3_planes = 12)
output = convolution_block(input)
```

### BinActive

The `BinActive` class is for calling binary activation method, which gives the sign of the input tensor as an activation.<br>
This is used in Binary Networks and XNOR Network.<br>

Usage:
```python
import torch
from niftytorch.Layers.Convolutional_Layers import BinActive
input = torch.ones(32,512,32,32,32)
activation = BinActive()
output = activation(input)
```

### BinConv3d

Parameters:

<ul>
<li>input_channels (int,required): The number of channels in the input tensor.
<li>output_channels (int,required): The number of channels in the output tensor.
<li>kernel_size (int,default = 3): Kernel size for the convolution filter.
<li>stride (int,default = 1): The number of strides in the convolutional filters.
<li>padding (int,default = 0): The padding which is to be done on the ends.
<li>groups (int,default = 1): Number of groups in the convolutional filters.
<li>dropout (int,default = 0): The dropout probability.
<li>Linear (bool,default = False): If True, instead of convolution.
</ul>
Usage:

```python
import torch
from niftytorch.Layers.Convolutional_Layers import BinConv3d
in_channels = 512
input = torch.ones(32,in_channels,32,32,32)
activation = BinConv3d(input_channels = in_channels,output_channels = 256,kernel_size = 3,stride = 2,padding = 1,groups = 1,dropout = 0.5,Linear = False)
output = activation(input)
```
