# Training

## Generic Training

This section describes the parameters common to all the models present in the Models class in NiftyTorch.

Parameters:
<ul>
<li>num_classes (int,required): The number of classes in a datasets.<br>
<li>in_channels (int,required): The number of channels in the input to the model.<br>
<li>data_folder (str,required): The path to the  directory which contains input data folder.<br>
<li>data_csv (str,required): The path to the csv containing the filename and it's corresponding label.<br>
<li>data_transforms (torchvision.transforms,required): The transformations from torchvision which is to be applied to the dataset.<br>
<li>filename_label (str,required): The label which used to identify the input image file name.<br>
<li>class_label (str,required): The label which is used to identify the class information for the corresponding filename.<br>
<li>learning_rate (float,default = 3e-4): The learning which is used to be for the optimizer.<br>
<li>step_size (int,default = 7): The step size to used for step based learning rate scheduler.<br>
<li>gamma (float,default = 0.2): The reduction factor to be used in step based learning rate scheduler.<br>
<li>cuda (str,default = None): The which GPU is to be used.<br>
<li>batch_size (int,default = 1): The number examples to be used in each gradient update.<br>
<li>image_scale (int,default = 128): The size of the image to be considerd for rescaling the image.<br>
<li>loss (torch.nn,default = nn.CrossEntropyLoss()): The loss function be used in the required task.<br>
<li>optimizer (torch.optim,default = optim.Adam): The optimizer which is used for updating weights.<br>
<li>device_ids (list,default = []): The list of GPUs to be considered for data parallelization.<br>
<li>l2 (float,default = 0): The l2 regularization coefficient.<br>
<li>experiment_name (string,default = None): The entire path to the tensorboard directory.<br>
</ul>

## Modules 

### AlexNet Training Parameters 

Parameters:
<ul>
<li>channels (list,default = [1,2,2,2,1]): A list containing the out_channels for each convolutional layer, it must be of size 5.
<li>kernel_size (list,default = [3,5,3,3,1]): A list containing the kernel size of each convolutional layer, it must be of size 5.
<li>strides (list,default = [1,2,1,1,1]): A list containing the stride at each convolutional layer, it must be of size 5.
<li>padding (list,default = [0,1,1,1,1]): A list containing the padding for each convolutional layer, it must be of size 5.
</ul>

Usage:

```python
import torch
from niftytorch.models.alexnet import train_alexnet
from torchvision import transforms

data_transforms = transforms.Compose([transforms.ToTensor()])
        
data_folder = "./data"
data_csv = "./data.csv"
train = train_alexnet()
train.set_params(num_classes = 2, in_channels = 1, data_folder = data_folder, data_csv = data_csv,channels = [1,2,4,2,1],kernel_size = [3,5,5,3,1],strides = [1,2,2,2,1],padding = [1,1,1,1,1], data_transforms = data_transforms, filename_label = 'Subject',class_label = 'Class',learning_rate = 3e-4,step_size = 7, gamma = 0.1, cuda = 'cuda:3',batch_size = 16,image_scale = 128)
train.train()
```

### VGGNet Training Parameters 

Parameters:
<ul>
<li>version (str,default = "A"): The version can be 'A','B','D' or 'E'.
<li>cfgs (list,default = same network parameters): A list containing the configurations.
</ul>

Usage:

```python
import torch
from niftytorch.models.vggnet import train_vggnet
from torchvision import transforms

data_transforms = transforms.Compose([transforms.ToTensor()])
        
data_folder = "./data"
data_csv = "./data.csv"
cfgs = {'B':[4, 'M', 8, 'M', 8, 8, 'M', 32, 32, 'M', 32, 64, 'M']}
train = train_vggnet()
train.set_params(num_classes = 2, in_channels = 1, data_folder = data_folder, data_csv = data_csv,version = "B", data_transforms = data_transforms, filename_label = 'Subject',class_label = 'Class',learning_rate = 3e-4,step_size = 7, gamma = 0.1, cuda = 'cuda:3',batch_size = 16,image_scale = 128,cfgs = cfgs)
train.train()
```

### ResNet Training Parameters 

Parameters:
<ul>
<li>block (default = BottleNeck): The type of network module to be used as a building block in the resnet.
<li>layers (list,default = [1,2,4,4]): The how many times does each block has to be repeated in the resnet.
<li>stride (list,default = [2,1,2,2,2]): The stride to be used in each building block of the resnet.
<li>channels (list,default = [64,128,256,512]): The number of channels to be maintained in each building block of the resnet.
</ul>

Usage:
```python
import torch
from niftytorch.models.resnet import train_resnet
from torchvision import transforms
from NiftyTorch.layers.layers import bottleneck
data_transforms = transforms.Compose([transforms.ToTensor()])
layers = [1,2,1,1,2]
stride = [1,1,1,1,1]
channels = [32,64,64,32,32]
data_folder = "./data"
data_csv = "./data.csv"
train = train_resnet()
train.set_params(num_classes = 2, in_channels = 1, data_folder = data_folder, data_csv = data_csv,block = block, data_transforms = data_transforms, filename_label = 'Subject',class_label = 'Class',layers = [1,2,4,4],stride = [2,1,2,2,2],channels = [64,128,256,512],learning_rate = 3e-4,step_size = 7, gamma = 0.1, cuda = 'cuda:3',batch_size = 16,image_scale = 128)
train.train()
```

### ShuffleNet Training Parameters 

Parameters:
<ul>
<li>groups (default = 2): number of groups to be used in grouped 1x1 convolutions in each ShuffleUnit.
<li>stage_repeats (list,default = [3,7,3]): The number of times each stage is repeated.
</ul>

Usage:
```python
import torch
from niftytorch.models.shufflenet import train_shufflenet
from torchvision import transforms
data_transforms = transforms.Compose([transforms.ToTensor()])
groups = 2
stage_repeats = [2,7,4]
data_folder = "./data"
data_csv = "./data.csv"
train = train_shufflenet()
train.set_params(num_classes = 2, in_channels = 1, data_folder = data_folder, data_csv = data_csv,groups = groups, data_transforms = data_transforms, filename_label = 'Subject',class_label = 'Class',stage_repeats = stage_repeats,learning_rate = 3e-4,step_size = 7, gamma = 0.1, cuda = 'cuda:3',batch_size = 16,image_scale = 128)
train.train()
```

### SqueezeNet Training Parameters

Parameters:
<ul>
<li>version (str,default = '1_0'): The version of squeezenet to be used.
</ul>

Usage:
```python
import torch
from niftytorch.models.squeezeNet import train_squeezenet
from torchvision import transforms
data_transforms = transforms.Compose([transforms.ToTensor()])
version = '1_1'
data_folder = "./data"
data_csv = "./data.csv"
train = train_squeezenet()
train.set_params(num_classes = 2, in_channels = 1, data_folder = data_folder, data_csv = data_csv,version = version, data_transforms = data_transforms, filename_label = 'Subject',class_label = 'Class',learning_rate = 3e-4,step_size = 7, gamma = 0.1, cuda = 'cuda:3',batch_size = 16,image_scale = 128)
train.train()
```

### XNOR NET

Parameters:
<ul>
<li>channels (list,default = [32, 96, 144, 144, 96, 7]): A list containing the out_channels for each convolutional layer, it must be of size 5.
<li>kernel_size (list,default = [11, 5, 3, 3, 3]): A list containing the kernel size of each convolutional layer, it must be of size 5.
<li>strides (list,default = [4, 1, 1, 1, 1]): A list containing the stride at each convolutional layer, it must be of size 5.
<li>padding (list,default = [0, 2, 1, 1, 1]): A list containing the padding for each convolutional layer, it must be of size 5.
<li>groups (list,default = [1, 1, 1, 1, 1]): A list containing the number of groups in convolution filters for each convolutional layer, it must be of size 5. 
</ul>

Usage:
```python
import torch
from niftytorch.models.xnornet import train_xnornet
from torchvision import transforms
data_transforms = transforms.Compose([transforms.ToTensor()])
version = '1_1'
data_folder = "./data"
data_csv = "./data.csv"
train = train_xnornet()
train.set_params(num_classes = 2, in_channels = 1, data_folder = data_folder, data_csv = data_csv,channels = [32,96,144,42,6],kernel_size = [3,5,5,3,1],strides = [1,2,1,2,1],padding = [1,1,0,1,1],groups = [1,2,2,1,1] data_transforms = data_transforms, filename_label = 'Subject',class_label = 'Class',learning_rate = 3e-4,step_size = 7, gamma = 0.1, cuda = 'cuda:3',batch_size = 16,image_scale = 128)
train.train()
```

## Hyperparameter Training

### Generic Hyperparameter Tuning parameters

For hyperparameter tuning we need to create a configuration dictionary where below parameters are the keys.  
These are generic parameters for hyperparameter tuning:

<ul>
<li>learning_rate (bool/float): If False, the hyperparameter tuning is considered for learning rate else for any other float value or True.
<li>lr_min (float): Minimum learning rate to be considered for tuning.
<li>lr_max (float): Maximum learning rate to be considered for tuning.
<li>batch_size (bool/int): If False, the hyperparameter tuning is considered for batch size else for any other int value or True. 
<li>data_folder (string): The path to the  directory which contains input data folder
<li>data_csv (string): The path to the csv containing the filename and it's corresponding label.
<li>gamma (float): The reduction factor to be used in step based learning rate scheduler.
<li>num_classes (int): The number of classes in a datasets.
<li>loss (bool): If False, the hyperparameter tuning is considered for loss else a single loss function is considered.
<li>step_size (float): The step size to used for step based learning rate scheduler
<li>loss_list (list): The list of losses to be considered for training.
<li>scheduler (bool): If False, the hyperparameter tuning is considered for scheduler else single scheduler function is considered.
<li>scheduler_list (list): The list of scheduler to be considered for hyperparameter tuning.
<li>optimizer (bool/nn.optim): If False, the hyperparameter tuning is considered for optimizer else a single optimizer function is considered. 
<li>opt_list (list): The list of optimizer to be considered for optimization.
<li>filename_label (str): The label which used to identify the input image file name.
<li>class_label (str): The label which is used to identify the class information for the corresponding filename.
<li>in_channels (int): The number of channels in the input to the model.
<li>num_workers (int): The threads to be considered while loading the data.
<li>image_scale (bool): The size of the image to be considerd for rescaling the image.
<li>image_scale_list (list): The list of the image sizes to be considered for hyperparameter tuning.
<li>device_ids (list): The list of devices to be considered for data parallelization.
<li>cuda (str): The GPU to be considered for loading data.
<li>l2 (float,default = 0): The l2 regularization coefficient.<br>
</ul>

### AlexNet Hyparameter Tuning

These are hyperparameters for AlexNet:

<ul>
<li>channels (int/bool): If False, the hyperparameter tuning is considered for channels.
<li>channels_1 (list): The list containing all values to be tested for channel 1.
<li>channels_2 (list): The list containing all values to be tested for channel 2.
<li>channels_3 (list): The list containing all values to be tested for channel 3.
<li>channels_4 (list): The list containing all values to be tested for channel 4.
<li>channels_5 (list): The list containing all values to be tested for channel 5.
<li>strides (bool): If False, the hyperparameter tuning is considered for strides.
<li>strides_1 (list): The list containing all values to be tested for strides 1.
<li>strides_2 (list): The list containing all values to be tested for strides 2.
<li>strides_3 (list): The list containing all values to be tested for strides 3.
<li>strides_4 (list): The list containing all values to be tested for strides 4.
<li>strides_5 (list): The list containing all values to be tested for strides 4.
<li>kernel_size (bool): If False, the hyperparameter tuning is considered for kernel size.
<li>kernel_size_1 (list): The list containing all values to be tested for kernel size 1.
<li>kernel_size_2 (list): The list containing all values to be tested for kernel size 2.
<li>kernel_size_3 (list): The list containing all values to be tested for kernel size 3.
<li>kernel_size_4 (list): The list containing all values to be tested for kernel size 4.
<li>kernel_size_5 (list): The list containing all values to be tested for kernel size 5.
<li>padding (bool): If False, the hyperparameter tuning is considered for padding.
<li>padding_1 (list): The list containing all values to be tested for padding 1.
<li>padding_2 (list): The list containing all values to be tested for padding 2.
<li>padding_3 (list): The list containing all values to be tested for padding 3.
<li>padding_4 (list): The list containing all values to be tested for padding 4.
<li>padding_5 (list): The list containing all values to be tested for padding 5.
</ul>

### ResNet Hyperparameter Tuning

These are hyperparameters for ResNet:

<ul>
<li>groups (bool): If True, the hyperparameter tuning is considered for groups. 
<li>groups_min (int): The minimum group value to be used for hyperparameter tuning.
<li>groups_max (int): The maximum group value to be used for hyperparameter tuning.
<li>block (bool): If True, the type of block in resent is considered for hyperparameter.
<li>block_list (list): The list containing different type of blocks to be considered for hyperparameter tuning. 
<li>norm_layer (bool): If True, the type of normalization layers in resent is considered for hyperparameter.
<li>norm_layer_list (list): The list containing different type of normalization layers to be considered for hyperparameter tuning.
<li>width_per_group (bool): If True, the number of groups per each layer in resent is considered for hyperparameter. 
<li>width_per_group_list (list): The list containing different types of groups.
</ul>

### ShuffleNet Hyperparameter Tuning

These are hyperparameters for ShuffleNet:

<ul>
<li>groups (bool): If True, the number of groups per each layer in resent is considered for hyperparameter.
<li>groups_min (int): The minimum group value to be used for hyperparameter tuning. 
<li>groups_max (int): The maximum group value to be used for hyperparameter tuning.
<li>stage_repeat (bool): If True, the number of stage_repeats in shufflenet is considered for hyperparameter.
<li>stage_repeat_1 (list): The list containing all the values stage_repeats_1.
<li>stage_repeat_2 (list): The list containing all the values stage_repeats_2.
<li>stage_repeat_3 (list): The list containing all the values stage_repeats_3.
</ul>

### XNOR NET Hyperparameter Tuning

<ul>
<li>channels (int/bool): If False, the hyperparameter tuning is considered for channels.
<li>channels_1 (list): The list containing all values to be tested for channel 1.
<li>channels_2 (list): The list containing all values to be tested for channel 2.
<li>channels_3 (list): The list containing all values to be tested for channel 3.
<li>channels_4 (list): The list containing all values to be tested for channel 4.
<li>channels_5 (list): The list containing all values to be tested for channel 5.
<li>strides (bool): If False, the hyperparameter tuning is considered for strides.
<li>strides_1 (list): The list containing all values to be tested for strides 1.
<li>strides_2 (list): The list containing all values to be tested for strides 2.
<li>strides_3 (list): The list containing all values to be tested for strides 3.
<li>strides_4 (list): The list containing all values to be tested for strides 4.
<li>strides_5 (list): The list containing all values to be tested for strides 4.
<li>kernel_size (bool): If False, the hyperparameter tuning is considered for kernel size.
<li>kernel_size_1 (list): The list containing all values to be tested for kernel size 1.
<li>kernel_size_2 (list): The list containing all values to be tested for kernel size 2.
<li>kernel_size_3 (list): The list containing all values to be tested for kernel size 3.
<li>kernel_size_4 (list): The list containing all values to be tested for kernel size 4.
<li>kernel_size_5 (list): The list containing all values to be tested for kernel size 5.
<li>padding (bool): If False, the hyperparameter tuning is considered for padding.
<li>padding_1 (list): The list containing all values to be tested for padding 1.
<li>padding_2 (list): The list containing all values to be tested for padding 2.
<li>padding_3 (list): The list containing all values to be tested for padding 3.
<li>padding_4 (list): The list containing all values to be tested for padding 4.
<li>padding_5 (list): The list containing all values to be tested for padding 5.
<li>groups (bool): If False, the hyperparameter tuning is considered for groups.
<li>groups_1 (list): The list containing all values to be tested for groups 1.
<li>groups_2 (list): The list containing all values to be tested for groups 2.
<li>groups_3 (list): The list containing all values to be tested for groups 3.
<li>groups_4 (list): The list containing all values to be tested for groups 4.
<li>groups_5 (list): The list containing all values to be tested for groups 5.
</ul>
