# [CIKM23] AmpliBias: Mitigating Dataset Bias through Bias Amplification in Few-shot Learning for Generative Models
Official PyTorch Implementation of [AmpliBias: Mitigating Dataset Bias through Bias Amplification in Few-shot Learning for Generative Models](https://dl.acm.org/doi/10.1145/3583780.3615184) (CIKM 2023) by Donggeun Ko et. a

### On-Going! Still fixing and revising code. Thank you for waiting :D 


## Environment Setting
python=3.8
torch=2.0.1 
torchvision=0.15.2

Used A100 GPU with CUDA version of 11.7. 
I tried with Torch version >=1.7.0 and it works perfectly fine.
The dependencies should match with [FastGAN-PyTorch](https://github.com/odegeasslbc/FastGAN-pytorch) when training FastGAN.

## 1. Description
The code is structured as follows:

* models.py : Model classifiers. For each dataset, we use different models. (CMNIST: 3-Layers MLP, Rest of the dataset: ResNet18)

* train_classifier.py : train code for the model classifiers. This is the first part where we intentionally train biased classifier with GCE loss.

* extract_bias.py: extract bias from the trained model classifier. This is the second part where we extract bias samples with high loss values from the trained model classifier

* 

## 2. Dataset
Dataset can be downloaded from the link below. 
The datasets are curated from [BiasEnsemble](https://github.com/kakaoenterprise/BiasEnsemble) by Jungsoo Lee et. al.
* [CMNIST](https://drive.google.com/file/d/1f4U7WPv0q_6TCilr4L1Ip-4WU8P5rXMq/view?usp=drive_link)

* [CIFAR10C](https://drive.google.com/file/d/1kOFjfhWRRzfgubCv5Ur9WFuT24qQNKn5/view?usp=drive_link)

* [BFFHQ](https://drive.google.com/file/d/1ZWWjXxcDVK_dATo3zbtHgYgEXn_NVrqm/view?usp=drive_link)

* [BAR](https://drive.google.com/file/d/1dCq6QWNSMvFED0PveyF6VleUXVxdu7xH/view?usp=drive_link)

## Install Dependencies
'''
pip install -r requirements.txt
'''

## 
