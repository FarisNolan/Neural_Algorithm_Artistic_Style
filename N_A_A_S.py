# -*- coding: utf-8 -*-
"""
Created on Thu Dec 27 08:33:31 2018

@author: Faris
"""
#-----TO DO-----
#   -Update Image directories


#-----IMPORTS AND DIRECTORIES-----
import time
import os 

image_dir = 'PATH TO IMAGES'
model_dir = 'PATH TO MODEL'

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

import torchvision
from torchvision import transforms

from PIL import Image
from collections import OrderedDict

import matplotlib.pyplot as plt

#------------------------------
#-----VGG MODEL DEFINITION-----
#------------------------------

#CAN RETURN OUTPUT FROM ANY LAYER
class VGG(nn.Module):
    def __init__(self, pool='max'):
        super(VGG, self).__init__()
        #CONV LAYERS
        self.conv1_1 = nn.Conv2d(3, 64, kernel_size = 3, padding = 1)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size = 3, padding = 1)
        
        self.conv2_1 = nn.Conv2d(64, 128, kernel_size = 3, padding = 1)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size = 3, padding = 1)
        
        self.conv3_1 = nn.Conv2d(128, 256, kernel_size = 3, padding = 1)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size = 3, padding = 1)
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size = 3, padding = 1)
        self.conv3_4 = nn.Conv2d(256, 256, kernel_size = 3, padding = 1)
        
        self.conv4_1 = nn.Conv2d(256, 512, kernel_size = 3, padding = 1)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size = 3, padding = 1)
        self.conv4_3 = nn.Conv2d(512, 512, kernel_size = 3, padding = 1)
        self.conv4_4 = nn.Conv2d(512, 512, kernel_size = 3, padding = 1)
        
        self.conv5_1 = nn.Conv2d(512, 512, kernel_size = 3, padding = 1)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size = 3, padding = 1)
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size = 3, padding = 1)
        self.conv5_4 = nn.Conv2d(512, 512, kernel_size = 3, padding = 1)
        
        #HANDLE POOLING OPTIONS
        #MAX POOLING
        if pool == 'max':
            self.pool1 = nn.MaxPool2d(kernel_size = 2, stride = 2)
            self.pool2 = nn.MaxPool2d(kernel_size = 2, stride = 2)
            self.pool3 = nn.MaxPool2d(kernel_size = 2, stride = 2)
            self.pool4 = nn.MaxPool2d(kernel_size = 2, stride = 2)
            self.pool5 = nn.MaxPool2d(kernel_size = 2, stride = 2)
        #AVERAGE POOLING
        elif pool == 'avg':
            self.pool1 = nn.AvgPool2d(kernel_size = 2, stride = 2)
            self.pool2 = nn.AvgPool2d(kernel_size = 2, stride = 2)
            self.pool3 = nn.AvgPool2d(kernel_size = 2, stride = 2)
            self.pool4 = nn.AvgPool2d(kernel_size = 2, stride = 2)
            self.pool5 = nn.AvgPool2d(kernel_size = 2, stride = 2)
            
        #FORWARD PROP
    def forward(self, x, out_keys):
        out = {}
        
        out['r11'] = F.relu(self.conv1_1(x))
        out['r12'] = F.relu(self.conv1_2(out['r11']))
        out['p1'] = self.pool1(out['r12'])
        
        out['r21'] = F.relu(self.conv2_1(out['p1']))
        out['r22'] = F.relu(self.conv2_2(out['r21']))
        out['p2'] = self.pool2(out['r22'])
        
        out['r31'] = F.relu(self.conv3_1(out['p2']))
        out['r32'] = F.relu(self.conv3_2(out['r31']))
        out['r33'] = F.relu(self.conv3_3(out['r32']))
        out['r34'] = F.relu(self.conv3_4(out['r33']))
        out['p3'] = self.pool3(out['r34'])
        
        out['r41'] = F.relu(self.conv4_1(out['p3']))
        out['r42'] = F.relu(self.conv4_2(out['r41']))
        out['r43'] = F.relu(self.conv4_3(out['r42']))
        out['r44'] = F.relu(self.conv4_4(out['r43']))
        out['p4'] = self.pool4(out['r44'])
        
        out['r51'] = F.relu(self.conv5_1(out['p4']))
        out['r52'] = F.relu(self.conv5_2(out['r51']))
        out['r53'] = F.relu(self.conv5_3(out['r52']))
        out['r54'] = F.relu(self.conv5_4(out['r53']))
        out['p5'] = self.pool5(out['r54'])
        
        
        #RETURN DESIRED ACTIVATIONs
        return [out[key] for key in out_keys]

#----------------------------------------------------    
#-----COMPUTING GRAM MATRIX AND GRAM MATRIX LOSS-----.0
#----------------------------------------------------
            
#GRAM MATRICES ARE USED TO MEASURE STYLE LOSS
#MATRIX
class GramMatrix(nn.Module):
    def forward(self, input):
        b, c, w, h = input.size()
        F = input.view(b, c, h * w)
        #COMPUTES GRAM MATRIX BY MULTIPLYING INPUT BY TRANPOSE OF ITSELF
        G = torch.bmm(F, F.transpose(1, 2))
        G.div_(h*w)
        return G

#LOSS
class GramMSELoss(nn.Module):
    def forward(self, input, target):
        out = nn.MSELoss()(GramMatrix()(input), target)
        return out
    
#--------------------------    
#-----IMAGE PROCESSING-----
#--------------------------
        
img_size = 512

#PRE-PROCESSING
prep = transforms.Compose([transforms.Scale(img_size),
                           transforms.ToTensor(),
                           transforms.Lambda(lambda x: x[torch.LongTensor([2, 1, 0])]), #CONVERT TO BGR FOR VGG NET
                           transforms.Normalize(mean = [0.40760392, 0.45795686, 0.48501961], std = [1, 1, 1]), #SUBTRACT IMAGENET MEAN
                           transforms.Lambda(lambda x: x.mul_(255)), #VGG WAS TRAINED WITH PIXEL VALUES 0-255
])

#POST PROCESSING A
postpa = transforms.Compose([transforms.Lambda(lambda x: x.mul_(1./255)),
                            transforms.Normalize(mean = [-0.40760392, -0.45795686, -0.48501961], std = [1, 1, 1]),
                            transforms.Lambda(lambda x: x[torch.LongTensor([2,1,0])]),    
        ])
    
#POST PROCESSING B
postpb = transforms.Compose([transforms.ToPILImage()])

#POST PROCESSING FUNCTION INCORPORATES A AND B, AND CLIPS PIXEL VALUES WHICH ARE OUT OF RANGE
def postp(tensor):
    t = postpa(tensor)
    t[t>1] = 1
    t[t<0] = 0
    img = postpb(t)
    return img

#---------------------------          
#-----PREPARING NETWORK-----
#--------------------------- 
    
vgg = VGG()

vgg.load_state_dict(torch.load(model_dir + 'vgg_conv_weights.pth'))            
for param in vgg.parameters():
    param.requires_grad = False
if torch.cuda.is_available():
    vgg.cuda()
    
#-----LOADING AND PREPARING IMAGES-----
img_dirs = [image_dir, image_dir]

#IMAGE LOADING ORDER: STYLE, CONTENT
img_names = ['style_pointillism_kingdom.jpg', 'content_hongkong.jpg']
imgs = [Image.open(img_dirs[i] + name) for i, name in enumerate(img_names)]
imgs_torch = [prep(img) for img in imgs]

#HANDLE CUDA
if torch.cuda.is_available():
    imgs_torch = [Variable(img.unsqueeze(0)).cuda() for img in imgs_torch]
else:
    imgs_torch = [Variable(img.unsqueeze(0)) for img in imgs_torch]
style_img, content_img = imgs_torch
for img in imgs_torch:
  print("Image size: ", img.size())

#SET UP IMAGE TO BE OPTIMIZED
#CAN BE INITIALIZED RANDOMLY OR AS A CLONE OF CONTENT IMAGE, AS DONE BELOW
opt_img = Variable(content_img.clone(), requires_grad = True)
print(content_img.size())
print(opt_img.size())

#DISPLAY IMAGES
for img in imgs:
    plt.grid(None)
    plt.imshow(img)
    plt.show()
    
#----------------------------
#-----SETUP FOR TRAINING-----
#----------------------------
#LAYERS FOR STYLE AND CONTENT LOSS
style_layers = ['r11', 'r12', 'r31', 'r41', 'r51']
content_layers = ['r42']
loss_layers = style_layers + content_layers

#CREATING LOSS FUNCTION
loss_fns = [GramMSELoss()] * len(style_layers) + [nn.MSELoss()] * len(content_layers)
if torch.cuda.is_available():
    loss_fns = [loss_fn.cuda() for loss_fn in loss_fns] 
    
#SETUP WEIGHTS FOR LOSS LAYERS
style_weights = [1e3/n**2 for n in [64, 128, 256, 512, 512]]
content_weights = [1e0]
weights = style_weights + content_weights

#CREATE OPTIMIZATION TARGETS
style_targets = [GramMatrix()(A).detach() for A in vgg(style_img, style_layers)]
content_targets = [A.detach() for A in vgg(content_img, content_layers)]
targets = style_targets + content_targets

#-----------------------
#-----TRAINING LOOP-----
#-----------------------
max_iter = 500
show_iter = 50
optimizer = optim.LBFGS([opt_img])
print(opt_img.size())
print(content_img.size())
n_iter = [0]

#ENTER LOOP
while n_iter[0] <= max_iter:
  
    def closure():
        optimizer.zero_grad()
        
        #FORWARD
        out = vgg(opt_img, loss_layers)
        
        #LOSS
        layer_losses = [weights[a] * loss_fns[a](A, targets[a]) for a,A in enumerate(out)]
        loss = sum(layer_losses)
        
        #BACKWARDS
        loss.backward()
        
        #TRACK PROGRESS
        n_iter[0] += 1
        if n_iter[0] % show_iter == (show_iter - 1):
            print('Iteration: %d,\tLoss: %f' % (n_iter[0] + 1, loss.data[0]))
            
        return loss
    
    optimizer.step(closure)
    
#-----------------
#-----RESULTS-----
#-----------------
print(float(opt_img.size(3)))
print(float(content_img.size(3)))
out_img = postp(opt_img.data[0].cpu().squeeze())
print(float(prep(out_img).size(2)))
plt.grid(None)
plt.imshow(out_img)
plt.gcf().set_size_inches(10, 10)
  

#<-------------------------------------------------------------------------------->
#<-------------------------------------------------------------------------------->
#<-------------------SECTION 2: PRODUCING HIGH RESOLUTION OUTPUT------------------->
#<-------------------------------------------------------------------------------->
#<-------------------------------------------------------------------------------->

#---------------------------
#-----HR PRE-PROCESSING-----
#---------------------------
img_size_hr = 800 #FOR 8GB GPU, CAN MAKE LARGER IF YOU HAVE MORE

prep_hr = transforms.Compose([
                              transforms.Scale(img_size_hr),
                              transforms.ToTensor(),
                              transforms.Lambda(lambda x: x[torch.LongTensor([2, 1, 0])]),
                              transforms.Normalize(mean=[0.40760392, 0.45795686, 0.48501961], std = [1, 1, 1]),
                              transforms.Lambda(lambda x: x.mul_(255))
])

#PREPARE CONTENT IMAGE
content_img = postp(content_img.data[0].cpu().squeeze())
content_img = prep_hr(content_img)


#IMAGES TORCH
imgs_torch = [prep_hr(imgs[0]), content_img]

if torch.cuda.is_available():
  imgs_torch = [Variable(img.unsqueeze(0).cuda()) for img in imgs_torch]
else:
  imgs_torch = [Variable(img.unsqueeze(0)) for img in imgs_torch]
  
style_img, content_img = imgs_torch

#CHANGE OPTIMIZATION IMAGE TO UPSIZED LOW RES VERSION
opt_img = prep_hr(out_img).unsqueeze(0)
opt_img = Variable(opt_img.type_as(content_img.data), requires_grad = True)

print(float(content_img.size(3)), float(opt_img.size(3)))
#-----------------------------------------
#-----PREPARE HR OPTIMIZATION TARGETS-----
#-----------------------------------------
style_targets = [GramMatrix()(A).detach() for A in vgg(style_img, style_layers)]
content_targets = [A.detach() for A in vgg(content_img, content_layers)]
targets = style_targets + content_targets

#------------------------------------------------------------
#-----RUNNING STYLE TRANSFER FOR HIGH RESOLUTION OUTPUT------
#------------------------------------------------------------
max_iter_hr = 200
optimizer = optim.LBFGS([opt_img])
n_iter = [0]

while n_iter[0] <= max_iter_hr:
  
  def closure():
    optimizer.zero_grad()
    out = vgg(opt_img, loss_layers)
    layer_losses = [weights[a] * loss_fns[a](A, targets[a]) for a,A in enumerate(out)]
    loss = sum(layer_losses)
    loss.backward()
    n_iter[0] += 1
    
    #DISPLAY LOSS
    if n_iter[0] % show_iter == (show_iter - 1):
      print('Iteration: %d,\tLoss: %f' % (n_iter[0] + 1, loss.data[0]))
      
    return loss
  
  optimizer.step(closure)
  
#---------------------  
#-----HR RESULTS------
#---------------------     
out_img_hr = postp(opt_img.data[0].cpu().squeeze())
plt.grid(None)
plt.imshow(out_img_hr)
plt.gcf().set_size_inches(10, 10)
    