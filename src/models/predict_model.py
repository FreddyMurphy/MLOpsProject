import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from src.data.dataloader import DIV2K


def save_model_output_figs(model):
    data = DIV2K('data/raw/DIV2K_valid_HR')
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
    plt.savefig("reports/figures/image_comparison.png")


if __name__ == '__main__':
    save_model_output_figs()
