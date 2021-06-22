import os

import matplotlib
import matplotlib.pyplot as plt
import torch
from hydra.utils import get_original_cwd
from torch.utils.data import DataLoader

from src.data.dataloader import DIV2K

# Needed to render on Linux
matplotlib.use('Agg')


def save_model_output_figs(model):
    data_path = os.path.join(get_original_cwd(),
                             'data/raw/DIV2K_valid_HR')
    data = DIV2K(data_path)
    dataloader = DataLoader(data, batch_size=1, num_workers=4)

    fig, axes = plt.subplots(3, 3)
    for idx, (high_res, low_res) in enumerate(dataloader):
        if (idx > 2):
            break
        print(high_res.shape)
        print(low_res.shape)

        high_res = high_res.squeeze(0)

        with torch.no_grad():
            upscaled = model(low_res).squeeze(0)
        low_res = low_res.squeeze(0)

        axes[idx, 0].imshow(high_res.permute(1, 2, 0))
        axes[idx, 1].imshow(upscaled.permute(1, 2, 0))
        axes[idx, 2].imshow(low_res.permute(1, 2, 0))

        axes[idx, 0].set_title('High-res')
        axes[idx, 0].axis('off')
        axes[idx, 1].set_title('Upscaled low-res')
        axes[idx, 1].axis('off')
        axes[idx, 2].set_title('Low res')
        axes[idx, 2].axis('off')

    fig.suptitle("Image comparison")
    plt.tight_layout()
    plt.savefig(os.path.join(get_original_cwd(),
                "reports/figures/image_comparison.png"))


if __name__ == '__main__':
    save_model_output_figs()
