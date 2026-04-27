import torch
import matplotlib.pyplot as plt
import numpy as np

path = 'src/exp/train/pretrained_weights/proposed_pnp_bandlimited_dau_10.pth'

params = torch.load(path)


red = np.array([params[f'alpha_red.{key}'].cpu().numpy() for key in range(10)])
lr = np.array([params[f'alpha_lr.{key}'].cpu().numpy() for key in range(10)])
pnp = np.array([params[f'alpha_pnp.{key}'].cpu().numpy() for key in range(10)])


# plt.plot(list(range(10)),np.power(10, red), label=r'$\alpha_{RED}$')
# plt.plot(list(range(10)),np.power(10, lr), label=r'$\alpha_{LR}$')
# plt.plot(list(range(10)),np.power(10, pnp), label=r'$\alpha_{PnP}$')

plt.plot(list(range(10)), red, label=r'$\alpha_{RED}$')
plt.plot(list(range(10)), lr, label=r'$\alpha_{LR}$')
plt.plot(list(range(10)), pnp, label=r'$\alpha_{PnP}$')

plt.legend()

plt.savefig('params.png')