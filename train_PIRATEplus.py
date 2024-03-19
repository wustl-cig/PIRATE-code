import torch
from torch import nn
import os
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
import math
import time
import SimpleITK as sitk
from model.base import *
from model.loss import *
from model.PIRATE import *
from model.PIRATEplus import *

def load_checkpoint(model, checkpoint_PATH, optimizer, device):
    checkpoint_PATH = checkpoint_PATH
    model_CKPT = torch.load(checkpoint_PATH, map_location=device)
    model.load_state_dict(model_CKPT['state_dict'], strict=False)
    optimizer.load_state_dict(model_CKPT['optimizer'])
    epoch = model_CKPT['epoch'] + 1
    print('loading checkpoint!')
    return model, optimizer, epoch

if __name__ == '__main__':
    image_path = "./data"
    model_path = "./pretrained_model/PIRATEplus/OASIS.pth.tar"
    save_path = "./pretrained_model/PIRATEplus"
    
    config_PIRATE = {
    "gamma_inti":5e5,
    "tau_inti":1e-7,
    "iteration":500,
    "image_shape":[160, 192, 224],
    "weight_grad":5e-1
    }
    
    config_PIRATEplus = {
    "max_iter":500,
    "tol":1e-3,
    "pre_train":True,
    "lambda_J":5,
    "lambda_df":1
    }
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    resize = ResizeTransform(1/2, 3)
    resize = resize.to(device)
    
    sim_func = NCC().to(device)

    denoiser = DnCNN()
    ForwardIteration = PIRATE(denoiser,config_PIRATE, device)
    PIRATEplus_model = PIRATEplus(ForwardIteration, config_PIRATEplus).to(device)
    
    if config_PIRATEplus["pre_train"] == True:
        optimizer = torch.optim.Adam(PIRATEplus_model.parameters(), lr=1e-6)
        PIRATEplus_model, optimizer, start_epoch= load_checkpoint(PIRATEplus_model, model_path, optimizer, device)
    else:
        optimizer = torch.optim.Adam(PIRATEplus_model.parameters(), lr=1e-6)
        start_epoch = 0
        
    for epoch in range(start_epoch, 60):
        PIRATEplus_model.train()
        loss_train = []
        loss_test = []
        
        pbar = tqdm(range(0,1))#replace by your own training size
        for step in pbar:
            
            #######replace by your own dataloader############
            moving = sitk.ReadImage('./data/moving.nii.gz')
            moving = sitk.GetArrayFromImage(moving)
            fixed = sitk.ReadImage('./data/fixed.nii.gz')
            fixed = sitk.GetArrayFromImage(fixed)
            #################################################
    
            moving = torch.from_numpy(moving).view(1, 1, moving.shape[-3], moving.shape[-2], moving.shape[-1]).to(device)
            fixed = torch.from_numpy(fixed).view(1, 1, fixed.shape[-3], fixed.shape[-2], fixed.shape[-1]).to(device)
        
            field = torch.zeros((1, 3, config_PIRATE['image_shape'][0]//2,config_PIRATE['image_shape'][1]//2, config_PIRATE['image_shape'][2]//2), requires_grad=True, device = device)
        
            field_hat, forward_iter, forward_res = PIRATEplus_model(field, moving, fixed)   
        
            field_full = resize(field_hat)
        
            transformer = SpatialTransformer(config_PIRATE['image_shape'])
            transformer = transformer.to(device)
            image_pred, field_with_grid = transformer(moving, field_full, return_phi=True)
        
            loss_j = config_PIRATEplus["lambda_J"] * neg_Jdet_loss(field_with_grid)
            loss_df = config_PIRATEplus["lambda_df"] * Grad().loss(field_full)
            loss = sim_func(fixed, image_pred) + loss_j + loss_df
            
            loss_train.append(loss.detach().to("cpu").item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        PIRATEplus_model.eval()
        with torch.no_grad(): 
            pbar = tqdm(range(0,1))#replace by your own training size
            for step in pbar:
                
                #######replace by your own dataloader############
                moving = sitk.ReadImage('./data/moving.nii.gz')
                moving = sitk.GetArrayFromImage(moving)
                fixed = sitk.ReadImage('./data/fixed.nii.gz')
                fixed = sitk.GetArrayFromImage(fixed)
                #################################################
                
                moving = torch.from_numpy(moving).view(1, 1, moving.shape[-3], moving.shape[-2], moving.shape[-1]).to(device)
                fixed = torch.from_numpy(fixed).view(1, 1, fixed.shape[-3], fixed.shape[-2], fixed.shape[-1]).to(device)
                
                field = torch.zeros((1, 3, config_PIRATE['image_shape'][0]//2,config_PIRATE['image_shape'][1]//2, config_PIRATE['image_shape'][2]//2), requires_grad=True, device = device)
        
                field_hat, forward_iter, forward_res = PIRATEplus_model(field, moving, fixed)   
        
                field_full = resize(field_hat)
            
                transformer = SpatialTransformer(config_PIRATE['image_shape'])
                transformer = transformer.to(device)
                image_pred, field_with_grid = transformer(moving, field_full, return_phi=True)
        
                loss_j = config_PIRATEplus["lambda_J"] * neg_Jdet_loss(field_with_grid)
                loss_df = config_PIRATEplus["lambda_df"] * Grad().loss(field_full)
                loss = sim_func(fixed, image_pred) + loss_j + loss_df
            
                loss_test.append(loss.detach().to("cpu").item())
                
        if epoch % 5 == 0:
            torch.save({'epoch': epoch, 'state_dict': PIRATEplus_model.state_dict(),
                    'optimizer': optimizer.state_dict()},
                    save_path + '/epoch_' + str(epoch) + '.pth.tar')
        
        epoch_info = 'Epoch %d/%d' % (epoch, 60)

        loss_info = 'train_loss: %.4e' % (np.mean(loss_train))
        test_info = 'test_loss: %.4e' % (np.mean(loss_test))
        print(' - '.join((epoch_info, loss_info,test_info)), flush=True)
                
