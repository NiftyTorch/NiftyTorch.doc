# Attention

The Attention can be used in the form of self-attention to identify important positional features or channel features. It can be also used **to add demographic data** into the model.</br>

## Modules

###Positional Attention 

The code below demonstrates how to use the Positional Attention Module (PAM).<br>

The object constructer for the Position Attention Module used to attend to different location specific features via aggreagation context.<br>

Parameters for Constructor:<br>

<ul>
<li> in_shape (int,required): the number of channels in the input tensor for PAM Module<br>
<li> reduction (int,default = 8): the compression in features channels to be done before computing the attention<br>
<li> query_conv_kernel (int, default = 1): The kernel size for convolutional filter applied in query features<br>
<li> key_conv_kernel (int, default = 1): The kernel size for convolutional filter applied in key features<br>
<li> value_conv_kernel (int, default = 1): The kernel size for convolutional filter applied in value features<br>
</ul>

Usage:

```python
from niftytorch.Attention.Attention import PAM_Module
PAM = PAM_Module(in_shape = 512,reduction = 8,query_conv_kernel = 3,key_conv_kernel = 3,value_conv_kernel = 3)
t = torch.rand(64,512,32,32)
out,attention = PAM(t)
print(out.shape)
>>> 64 x 512 x 32 x 32
print(attention.shape)
>>> 64 x 1024 x 1024
```

The PAM returns two tensors where the first tensor is the output from positional attention and the second is the attention map.<br>

###Channel Attention 

The code below demonstrates how to use the Channel Attention Module (CAM).<br>

The object constructer for the Channel Attention Module used to attend to different channel specific features.<br>

Parameters for Constructor:<br>

<ul>
<li> in_shape (int,required): the number of channels in the input tensor for CAM Module<br>
</ul>

Usage:

```python
from niftytorch.Attention.Attention import CAM_Module
PAM = CAM_Module(512)
t = torch.rand(64,512,32,32)
out,attention = CAM(t)
print(out.shape)
>>> 64 x 512 x 32 x 32
print(attention.shape)
>>> 64 x 512 x 512
```

The CAM returns two tensors where the first tensor is the output from channel-wise attention and the second is the attention map.<br>


