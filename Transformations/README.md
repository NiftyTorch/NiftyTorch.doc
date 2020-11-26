# Transformations

## Modules

### Add Noise

The `Add_Noise` is often used for data augmentation and sometimes as regularization.<br>
The function adds gaussian centered around the given mean and standard deviation to the input.<br>

Parameters:
inputs (Torch.Tensor,required): The input tensor which has to be transformed.
mean (Torch.Tensor,default = None): The mean of gaussian noise to be added as a tensor.
std (Torch.Tensor,default = None): The std of gaussian noise to be added as a tensor.

Usage:

```python
import torch
input = torch.ones(64,512,32,32,32)
from niftytorch.Transformations.Transformations import Add_Noise
output = Add_Noise(input,mean = torch.zeros(input.shape),std = torch.eye(input.shape))
```

### Rotate 90

Rotate the input tensor by 90.

Parameters:
inputs (torch.Tensor,required): The input tensor which needs to rotated.

Usage:

```python
import torch
input = torch.ones(64,512,32,32,32)
from niftytorch.Transformations.Transformations import Rotate_90
output = Rotate_90(input)
```

### Rotate 180

Rotate the input tensor by 180.

Parameters:
inputs (torch.Tensor,required): The input tensor which needs to rotated.

Usage:

```python
import torch
input = torch.ones(64,512,32,32,32)
from niftytorch.Transformations.Transformations import Rotate_180
output = Rotate_180(input)
```

### Rotate 270

Rotate the input tensor by 270.

Parameters:
inputs (torch.Tensor,required): The input tensor which needs to rotated.

Usage:

```python
import torch
input = torch.ones(64,512,32,32,32)
from niftytorch.Transformations.Transformations import Rotate_270
output = Rotate_270(input)
```

### Random_Segmentation_Crop

The `Random_Segmentation_Crop` is often used with segmentation.<br>
The idea behind random crop is select regions around lesions and use only this region to train the network.

Parameters:
input: the input 3D image.
mask: the mask of the lesion region.
context: the region around the lesion to be considered for crop.

Usage:

```python
import torch
input = torch.ones(64,512,32,32,32)
mask = torch.zeros(64,512,32,32,32)
context = 32
from niftytorch.Transformations.Transformations import Random_Segmentation_Crop
input,mask = Random_Segmentation_Crop(input,mask,context)
```

### Resize

The `Resize` module is used to resize the input tensor to the given size.

Parameters:
input (torch.Tensor,required): the input 3D image.
mask (torch.Tensor,default = None): the mask of the lesion region.
common: the size to which the tensor is to be resized.

Usage:

```python
import torch
input = torch.ones(64,512,32,32,32)
mask = torch.zeros(64,512,32,32,32)
context = 32
from niftytorch.Transformations.Transformations import Resize
input,mask = Resize(input,mask,context)
```
