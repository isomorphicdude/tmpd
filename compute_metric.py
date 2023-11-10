from pathlib import Path
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error as mse
from tqdm import tqdm

import matplotlib.pyplot as plt
import lpips
import numpy as np
import torch


device = 'cuda:0'
# loss_fn_vgg = lpips.LPIPS(net='vgg').to(device)
loss_fn_vgg = lpips.LPIPS(net='vgg')

task = 'SR'
factor = 4
sigma = 0.1
scale = 1.0  # I think that this is the step size scaling


# Ground truth
# label_root = Path(f'/media/harry/tomo/FFHQ/256_1000')
label_root = Path('/Users/ben/sync/ddrm-exp-datasets/ood_celeba/0')

# Delta must be alternative method?
# delta_recon_root = Path(f'./results/{task}/ffhq/{factor}/{sigma}/ps/{scale}/recon')
# Normal must be their method?
# normal_recon_root = Path(f'./results/{task}/ffhq/{factor}/{sigma}/ps+/{scale}/recon')

delta_recon_root = Path('/Users/ben/sync/ddrm-exp-datasets/ood_bedroom/0')
normal_recon_root = Path('/Users/ben/sync/ddrm-exp-datasets/ood_celeba/0')

psnr_delta_list = []
psnr_normal_list = []
ssim_delta_list = []
ssim_normal_list = []
mse_delta_list = []
mse_normal_list = []
lpips_delta_list = []
lpips_normal_list = []
# for idx in tqdm(range(150)):
for idx in tqdm(range(10)):
    # fname = str(idx).zfill(5)
    fname = str(idx)

    print(fname)
    print(label_root / f'orig_{fname}.png')
    # label = plt.imread(label_root / f'{fname}.png')[:, :, :3]
    # delta_recon = plt.imread(delta_recon_root / f'{fname}.png')[:, :, :3]
    # normal_recon = plt.imread(normal_recon_root / f'{fname}.png')[:, :, :3]
    label = plt.imread(label_root / f'orig_{fname}.png')[:, :, :3]
    delta_recon = plt.imread(delta_recon_root / f'orig_{fname}.png')[:, :, :3]
    normal_recon = plt.imread(normal_recon_root / f'orig_{fname}.png')[:, :, :3]

    psnr_delta = psnr(label, delta_recon)
    psnr_normal = psnr(label, normal_recon)
    ssim_delta = ssim(label, delta_recon, data_range=delta_recon.max() - delta_recon.min(), channel_axis=2)
    ssim_normal = ssim(label, normal_recon, data_range=normal_recon.max() - normal_recon.min(), channel_axis=2)
    mse_delta = mse(label, delta_recon)
    mse_normal = mse(label, normal_recon)

    psnr_delta_list.append(psnr_delta)
    psnr_normal_list.append(psnr_normal)
    ssim_delta_list.append(ssim_delta)
    ssim_normal_list.append(ssim_normal)
    mse_delta_list.append(mse_delta)
    mse_normal_list.append(mse_normal)

    # Pre-processing for lpips metric
    # delta_recon = torch.from_numpy(delta_recon).permute(2, 0, 1).to(device)
    delta_recon = torch.from_numpy(delta_recon).permute(2, 0, 1)
    # normal_recon = torch.from_numpy(normal_recon).permute(2, 0, 1).to(device)
    normal_recon = torch.from_numpy(normal_recon).permute(2, 0, 1)
    # label = torch.from_numpy(label).permute(2, 0, 1).to(device)
    label = torch.from_numpy(label).permute(2, 0, 1)

    delta_recon = delta_recon.view(1, 3, 256, 256) * 2. - 1.
    normal_recon = normal_recon.view(1, 3, 256, 256) * 2. - 1.
    label = label.view(1, 3, 256, 256) * 2. - 1.

    delta_d = loss_fn_vgg(delta_recon, label)
    normal_d = loss_fn_vgg(normal_recon, label)

    # lpips_delta_list.append(delta_d)
    # lpips_normal_list.append(normal_d)
    lpips_delta_list.append(delta_d.detach().cpu().numpy().flatten()[0])
    lpips_normal_list.append(normal_d.detach().cpu().numpy().flatten()[0])

psnr_delta_avg = sum(psnr_delta_list) / len(psnr_delta_list)
ssim_delta_avg = sum(ssim_delta_list) / len(ssim_delta_list)
mse_delta_avg = sum(psnr_delta_list) / len(psnr_delta_list)
lpips_delta_avg = sum(lpips_delta_list) / len(lpips_delta_list)

psnr_normal_avg = sum(psnr_normal_list) / len(psnr_normal_list)
ssim_normal_avg = sum(ssim_normal_list) / len(ssim_normal_list)
mse_normal_avg = sum(mse_normal_list) / len(mse_normal_list)
lpips_normal_avg = sum(lpips_normal_list) / len(lpips_normal_list)

print(f'Delta PSNR: {psnr_delta_avg}')
print(f'Delta SSIM: {ssim_delta_avg}')
print(f'Delta MSE: {mse_delta_avg}')
print(f'Delta LPIPS: {lpips_delta_avg}')

print(f'Normal PSNR: {psnr_normal_avg}')
print(f'Normal SSIM: {ssim_normal_avg}')
print(f'Normal MSE: {mse_normal_avg}')
print(f'Normal LPIPS: {lpips_normal_avg}')
