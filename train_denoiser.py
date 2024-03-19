import os
import torch
from tqdm import tqdm
import numpy as np
import h5py
from model.base import *
from model.loss import *

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
    model_path = "./pretrained_model/AWGN_denoiser/OASIS.pth.tar"
    save_path = "./pretrained_model/AWGN_denoiser"
    
    config_denoisesr = {
    "field_shape":[3, 80, 96, 112],
    "sigma": 1,
    "pretrain":False
    }
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if config_denoisesr["pretrain"] == True:
        model = DnCNN()
        model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)
        model, optimizer, start_epoch= load_checkpoint(model, save_path, optimizer, device)
    else:
        model = DnCNN()
        model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)
        start_epoch = 0
    
    epoch_total_loss = []
    epoch_test_loss = []
    for epoch in range(start_epoch, 400):
        model.train()
        pbar = tqdm(range(0,1))#replace by your own training size
        for step in pbar:
            
            #######replace by your own dataloader############
            with h5py.File("./data/field.h5py", "r") as f:
                field = np.array(f["fieldData"],dtype = 'float32')
            field = torch.from_numpy(field)
            #################################################

            field = field.to(device)
            noised_field = field + (config_denoisesr["sigma"]**2)*torch.randn(3, field.shape[-3],field.shape[-2],field.shape[-1]).to(device) 
        
            noise = model(noised_field)

            loss_function = torch.nn.MSELoss()
            loss = loss_function(noised_field-noise,field)
            
            epoch_total_loss.append(loss.detach().to("cpu").item())
        
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            pbar = tqdm(range(0,1))#replace by your own inference size
            for step in pbar:
                
                #######replace by your own dataloader############
                with h5py.File("./data/field.h5py", "r") as f:
                    field = np.array(f["fieldData"],dtype = 'float32')
                field = torch.from_numpy(field)
                #################################################

                field = field.to(device)
                noised_field = field + (config_denoisesr["sigma"]**2)*torch.randn(3, field.shape[-3],field.shape[-2],field.shape[-1]).to(device) 
            
                noise = model(noised_field)

                loss_function = torch.nn.MSELoss()
                loss = loss_function(noised_field-noise,field)

                epoch_test_loss.append(loss.detach().to("cpu").item())

        if epoch % 50 == 0:
            torch.save({'epoch': epoch, 'state_dict': model.state_dict(),
                            'optimizer': optimizer.state_dict()},
                           save_path + '/epoch_' + str(epoch) + '.pth.tar')
        epoch_info = 'Epoch %d/%d' % (epoch + 1, 400)
        loss_info = 'loss: %.4e' % np.mean(epoch_total_loss)
        test_info = 'test_loss: %.4e' % np.mean(epoch_test_loss)

        print(' - '.join((epoch_info, loss_info,test_info)), flush=True)