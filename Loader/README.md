# Loader

The Loader class is used to Load data for various types of tasks such as classification, image generation and image segmentation.<br>
The Loader class has `Classification DataLoader` for classification tasks and `Segmentation DataLoader` for image-to-image (image generation) and image-to-labels (segmentation) tasks.

## Classification DataLoader

For `Classification DataLoader` the class label can be provided to the model in the form of a csv file, which contains a *filename label* and *classname label*.<br>
The input file is written as **input_image.nii.gz**<br>
The output file is written as **output_labels.csv**<br>

Parameters:
<ul>
<li>root (str,required): This is path to the directory where the data is stored
<li>csv_dir (str,required): This is the csv file which contains the subject and its corresponding label
<li>transform (torchvision.transforms,required) : The transforms here is the transforms from torchvision library
<li>target_transform (torchvision.transforms,required): The target tranforms is the same as transforms parameter but the transformation is done on the test data
<li>loader (function,required): The type of loader to be used, current loader uses nipy.
<li>is_valid_file (function,required): List of type of files to be considered to be loaded as an input data. 
<li>filename_label (str,required): The filename label in the CSV file.
<li>class_label (str,required): The class label in the CSV file.
<li>file_type (tuple,default = ('nii.gz','.nii')): The file types to be considered as an input for the model. The resulting files are stacked along channels in the input tensor.
<li>common (int,default = 64): The common is the size of the 3D Tensor.
<li>demographic (list,default = []): The list demographic factors to be used for classification.
</ul>

Usage:

```python
from niftytorch.loader.dataloader import ImageFolder
from torchvision import transforms
data_transforms = transforms.Compose([transforms.ToTensor()])
data_folder = "../data/train/"
data_csv = "../data.csv"
filename_label = "Subject"
class_label = "labels"
image_scale = 32
file_type = ('t1w.nii.gz','flair.nii.gz')
demographic = ['factor1','factor2']
image_datasets = ImageFolder('classification',root = data,data_csv = data_csv,transforms = data_transforms,target_transforms = data_transforms,loader = ,filename_label = filename_label,class_label = class_label,common = image_scale,file_type = file_type,demographic = demographic)
```

## Segmentation DataLoader

The `Segmentation DataLoader` is used for image segmentation and image generation.<br>
The input file is written as **input_image.nii.gz**<br>
The output file is written as **seg.nii.gz**<br>

Parameters:
<ul>
<li>root (str,required): This is path to the directory where the data is stored.
<li>transform (torchvision.transforms,required) : The transforms here is the transforms from torchvision library
<li>target_transform (torchvision.transforms,required): The target tranforms is the same as transforms parameter but the transformation is done on the test data
<li>loader (function,required): The type of loader to be used, current loader uses nipy.
<li>is_valid_file (function,required): List of type of files to be considered to be loaded as an input data.
<li>common (int,default = 64): The common is the size of the 3D Tensor.
</ul>

Usage:

```python
from niftytorch.Loader.dataloader import ImageFolder
from torchvision import transforms
data_transforms = transforms.Compose([transforms.ToTensor()])
data_folder = "../data/train/"
image_scale = 32
image_datasets = ImageFolder('segmentation',root = data,data_csv = data_csv,transforms = data_transforms,target_transforms = data_transforms,loader = ,filename_label = filename_label,class_label = class_label,common = image_scale)
```

## Paired DataLoader

For `Paired DataLoader` the class label can be provided to the model in the form of a csv file, which contains a *filename label* and *classname label*.<br>
The input file is written as **input_image.nii.gz**<br>
The output file is written as **output_labels.csv**<br>

Parameters:
<ul>
<li>root (str,required): This is path to the directory where the data is stored.
<li>transform (torchvision.transforms,required) : The transforms here is the transforms from torchvision library
<li>target_transform (torchvision.transforms,required): The target tranforms is the same as transforms parameter but the transformation is done on the test data
<li>loader (function,required): The type of loader to be used, current loader uses nipy.
<li>is_valid_file (function,required): List of type of files to be considered to be loaded as an input data.
<li>common (int,default = 64): The common is the size of the 3D Tensor.
<li>negative_examples (int, default = None): The number of negative example to be considered for each positive examples 
</ul>

Usage:

```python
from niftytorch.Loader.dataloader import ImageFolder
from torchvision import transforms
data_transforms = transforms.Compose([transforms.ToTensor()])
data_folder = "../data/train/"
image_scale = 32
image_datasets = ImageFolder('paired',root = data,data_csv = data_csv,transforms = data_transforms,target_transforms = data_transforms,loader = ,filename_label = filename_label,class_label = class_label,common = image_scale,negative_examples = 4)
```


