import torch
from src.data.dataloader import DIV2K
from torch.utils.data import DataLoader

div2k_train = DIV2K('data/raw/DIV2K_train_HR')
div2k_test = DIV2K('data/raw/DIV2K_valid_HR')

class TestData:

    # Test for checking the length of the training dataset
    def test_len_of_train(self):
        assert len(div2k_train) == 800

    # Test for checking the length of the test dataset
    def test_len_of_test(self):
        assert len(div2k_test) == 100

    # Test for checking that each index contains a low and high res image in train
    def test_shape_of_images_train(self):
        dataloader = DataLoader(div2k_train, batch_size=1, num_workers=4)
        for index, images in enumerate(dataloader):
            assert len(images) == 2

    # Test for checking that each index contains a low and high res image in test
    def test_shape_of_images(self):
        dataloader = DataLoader(div2k_test, batch_size=1, num_workers=4)
        for index, images in enumerate(dataloader):
            assert len(images) == 2


