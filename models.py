import os, cv2, json, random, math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from glob import glob
from tqdm import tqdm
from PIL import Image, ImageDraw

import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision
import torchvision.transforms as T

# from stylegan2_model import StyledConv, Blur, EqualLinear, EqualConv2d, ScaledLeakyReLU
# from op import FusedLeakyReLU

class MLP(nn.Module):
    def __init__(self, num_classes = 10):
        super(MLP, self).__init__()
        self.feature = nn.Sequential(
            nn.Linear(3*28*28, 100), # image size is 28 x 28 for cmnist
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, 16),
            nn.ReLU()
        )
        self.classifier = nn.Linear(16, num_classes)

    def forward(self, x, return_feat=False):
        x = x.view(x.size(0), -1) / 255
        feat = x = self.feature(x)
        x = self.classifier(x)

        if return_feat:
            return x, feat
        else:
            return x

class MLP_DisEnt(nn.Module):
    def __init__(self, num_classes = 10):
        super(MLP_DisEnt, self).__init__()
        self.feature = nn.Sequential(
            nn.Linear(3 * 28*28, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, 16),
            nn.ReLU()
        )
        self.fc = nn.Linear(32, num_classes)

    def forward(self, x, return_feat=False):
        x = x.view(x.size(0), -1) / 255
        feat = x = self.feature(x)
        x = self.fc(x)

        if return_feat:
            return x, feat
        else:
            return x

def call_by_name(args):
    if args.exp == 'cmnist' or args.exp == 'new-cmnist':
        if args.etc == 'vanilla':
            model = MLP(args.num_classes)
            return model

        elif args.etc == 'LfF':
            model_b = MLP(args.num_classes)
            model_d = MLP(args.num_classes)
            return model_b, model_d

        elif args.etc == 'DisEnt':
            model_b = MLP_DisEnt(args.num_classes)
            model_d = MLP_DisEnt(args.num_classes)
            return model_b, model_d

    else:
        if args.etc == 'vanilla':
            if args.exp == 'bar' or args.exp == 'dog_and_cats':
                model = torchvision.models.resnet18(weights='DEFAULT')
                model = torchvision.models.resnet18(pretrained=True)

                model.fc = nn.Linear(512, args.num_classes)
            else:
                model = torchvision.models.resnet18(pretrained=False)
                model.fc = nn.Linear(512, args.num_classes)    
            # model = MLP(args.num_classes)
            return model

        # if args.etc == 'LfF':
        #     # model_b = torchvision.models.resnet18(pretrained=args.pretrained)
        #     model_b = torchvision.models.resnet18(pretrained=args.pretrained)
        #     model_b.fc = nn.Linear(512, args.num_classes)
            
        #     # model_d = torchvision.models.resnet18(pretrained=args.pretrained)
        #     model_d = torchvision.models.resnet18(pretrained=args.pretrained)
        #     model_d.fc = nn.Linear(512, args.num_classes)
        #     return model_b, model_d
        
        # elif args.etc == 'DisEnt':
        #     # model_b = torchvision.models.resnet18(pretrained=args.pretrained)
        #     model_b = torchvision.models.resnet18(pretrained=args.pretrained)
        #     model_b.fc = nn.Linear(1024, args.num_classes)

        #     # model_d = torchvision.models.resnet18(pretrained=args.pretrained)
        #     model_d = torchvision.models.resnet18(pretrained=args.pretrained)
        #     model_d.fc = nn.Linear(1024, args.num_classes)
        #     return model_b, model_d
        
        # elif args.etc == 'BiaSwap':
        #     E = Encoder(args.channel)
        #     G = Generator(args.channel)
        #     D = Discriminator(args.size, channel_multiplier=args.channel_multiplier)
        #     CD = CooccurDiscriminator(args.channel)
        #     return E,G,D,CD
