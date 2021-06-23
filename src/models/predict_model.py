import os

import matplotlib
import matplotlib.pyplot as plt
import torch
from hydra.utils import get_original_cwd
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image

# Needed to render on Linux
matplotlib.use('Agg')


def save_model_output_figs(model, div2k):
    # data_path = os.path.join(get_original_cwd(), 'data/raw/DIV2K_valid_HR')
    # data = DIV2K(data_path)
    # dataloader = DataLoader(data, batch_size=1, num_workers=4)

    dataloader = div2k.test_dataloader()
    print('size:', len(dataloader))

    num_images = 3

    if (num_images > 5):
        num_images = 5

    fig, axes = plt.subplots(num_images, 3, figsize=(15, 15))
    for idx, (high_res, low_res) in enumerate(dataloader):
        for i in range(num_images):
            high_tensor = high_res[i, ...]

            with torch.no_grad():
                upscaled = model(low_res[i, ...].unsqueeze_(0))
            low_tensor = low_res[i, ...]

            upscaled = upscaled.squeeze(0)

            axes[i, 0].imshow(high_tensor.permute(1, 2, 0))
            axes[i, 1].imshow(upscaled.permute(1, 2, 0))
            axes[i, 2].imshow(low_tensor.permute(1, 2, 0))

            axes[i, 0].axis('off')
            axes[i, 1].axis('off')
            axes[i, 2].axis('off')

            axes[i, 0].set_xticklabels([])
            axes[i, 0].set_yticklabels([])
            axes[i, 1].set_xticklabels([])
            axes[i, 1].set_yticklabels([])
            axes[i, 2].set_xticklabels([])
            axes[i, 2].set_yticklabels([])

        axes[0, 0].set_title('High-res')
        axes[0, 1].set_title('Upscaled low-res')
        axes[0, 2].set_title('Low-res')

    fig.suptitle("Image comparison")
    plt.tight_layout()
    plt.savefig(
        os.path.join(get_original_cwd(),
                     "reports/figures/image_comparison_double.png"))


def infer_files(model, data_path):

    file_names = os.listdir(data_path)

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output_dir = os.path.join(data_path, 'upscaled')
    os.makedirs(output_dir, exist_ok=True)

    print(torch.cuda.memory_allocated(0))
    model = model.to(DEVICE)
    print(torch.cuda.memory_allocated(0))

    print('Starting inference')
    # batch_size=1 since images might not be same size
    for idx, file_name in enumerate(file_names):
        img_path = os.path.join(data_path, file_name)
        if os.path.isdir(img_path):
            continue

        with Image.open(img_path) as img:
            img = img.convert("RGB")

        img = transforms.ToTensor()(img)
        img = img.unsqueeze_(0)
        img = img.to(DEVICE)
        print('inferring index:', idx)

        with torch.no_grad():
            upscaled_tensor = model(img)

        upscaled_tensor = upscaled_tensor.to('cpu')
        img = img.to('cpu')
        image_path = os.path.join(output_dir, file_name)
        save_image(upscaled_tensor, image_path)
        del upscaled_tensor


if __name__ == '__main__':
    save_model_output_figs()
