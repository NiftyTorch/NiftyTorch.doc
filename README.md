 ![Logo](img/NiftyTorchLogo_1.png)

**NiftyTorch is a Python API for deploying deep neural networks for Neuroimaging research.**

# Motivation
The motivation behind the development of such a library is that there is no centralized tool for deployig 3D deep learning for neuroimaing. In addition, most of the existing tools require expert technical know-how in deep learning or programming, creating a barrier for entry. The goal is to provide a one stop API using which the users can perform classification tasks, Segmentation tasks and Image Generation tasks. The intended audience are the members of neuroimaging who would like to explore deep learning but have no background in coding.<br>

For demos, see [this folder](https://github.com/NiftyTorch/NiftyTorch.v.0.1/tree/master/Demo).

## Highlighted Features 
<ul>
<li>Pytorch Embedded End to End data-loading pipeline
<li>Built-in Attention module to demographic data and other modalities or mask
<li>Additional Loss Functions (not in PyTorch) incorporated to easily use with any network use
<li>Automatic Hyperparameter tuning for each network
<li>Easily Customizable to use your own network
<li>Multi-scale Training
<li>Built-in classification networks
<li>Built-in CNNs units such as Bottleneck, BinaryActivation
</ul>

## Features
<ul>
<li>Data Loader
<ul>
<li>Classification DataLoader for nifty files
<li>Segmentation DataLoader for nifty files
</ul>
<li>Loss Functions
<ul>
<li>Cross Entropy
<li>Focal Loss
<li>Dice Loss
<li>Focal Dice Loss
<li>Tversky Loss
<li>Lovarsz Softmax
<li>Triplet Loss
<li>Contrastive Loss
</ul>
<li>Support for including demographic information using attention
<ul>
<li>Position Attention
<li>Channel Attention
</ul>
<li>Convolutional Neural Network Units (Layers)
<ul>
<li>BottleNeck Unit
<li>Fire Unit
<li>Shuffle Unit
<li>Binary Activation
<li>Binary Convolution
</ul>
<li>Data Augmentation (Transformations)
<ul>
<li>Noise Addition
<li>Rotation
<li>Random Segmentation Crop
<li>Resize
</ul>
<li>Models
<ul>
<li>AlexNet
<li>VGGNet
<li>ResNet
<li>ShuffleNet
<li>SqueezeNet
<li>XNORNet
</ul>
<li>Training
<ul>
<li>Data level Parallelization
<li>Multi Scale Training
<li>Hyperparameter Training
</ul>
</ul>


## Installation

NiftyTorch can be installed using:  
```python
pip install niftytorch==0.1.1 --extra-index-url=https://pypi.org/simple/
```
If you encounter problem, check dependencies. NiftyTorch requires `torch==1.4.0`, `torchvision==0.5.0` and `optuna==1.4.0`. We also noted in some machines latest Numpy generates error (if so, change to an older version, such as `numpy-1.16.4` using `pip install numpy==1.16.4`). For a complete demo on how to set up the requirement and getting started see **Getting Started** notebook in the [Demo](https://github.com/NiftyTorch/NiftyTorch.v.0.1/tree/master/Demo) folder.

## Resources

- For **Tutorials and Demos**, please visit [Demo Repository](https://github.com/NiftyTorch/NiftyTorch.v.0.1/tree/master/Demo). For **Documentation**, please visit [niftytorch.github.io](http://niftytorch.github.io/doc/).  
- For **Announcements and News**, follow us on Twitter [@NiftyTorch](https://twitter.com/NiftyTorch).  
- For Please submit your **questions** and **suggestions** via `niftytorch @ gmail.com`. We appreciate your constructive inputs. 

## Developers
**Adithya Subramanian** and **Farshid Sepehrband**  
INI Microstructural imaging Group ([IMG](https://www.ini.usc.edu/IMG/))  
Mark and Mary Stevens Neuroimaging and Informatics Institute ([INI](https://www.ini.usc.edu/))  
Keck School of Medicine of **USC**

## Other Contributors
**Sankareswari Govindarajan** and **Haoyu Lan**  
INI Microstructural imaging Group ([IMG](https://www.ini.usc.edu/IMG/))  

## Acknowledgement

NiftyTorch would not be possible without liberal imports of the excellent [pytorch](https://pytorch.org), [torchvision](https://pytorch.org/docs/stable/torchvision/index.html), [nipy](https://nipy.org), [numpy](https://numpy.org), [pandas](https://pandas.pydata.org), [matplotlib](https://matplotlib.org) and [optuna](https://github.com/optuna/optuna) libraries. 
