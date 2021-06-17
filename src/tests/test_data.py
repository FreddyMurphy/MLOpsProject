import torch
from torch.utils.data import DataLoader

from src.data.dataloader import DIV2K, DIV2KDataModule

div2k_train = DIV2K('data/raw/DIV2K_train_HR')
div2k_test = DIV2K('data/raw/DIV2K_valid_HR')
data_module = DIV2KDataModule(batch_size=1, num_workers=4)
data_module.setup()


class TestData:

    # Test for checking the length of the training dataset
    def test_len_of_train(self):
        assert len(div2k_train) == 800

    # Test for checking the length of the test dataset
    def test_len_of_test(self):
        assert len(div2k_test) == 100

    # Test for checking that each index contains a low and high res image in
    # train
    def test_shape_of_images_train(self):
        dataloader = data_module.train_dataloader()
        for index, images in enumerate(dataloader):
            assert len(images) == 2

    # Test for checking that all the indices in div2k_train contains two
    # tensors
    def test_train_index_contains_two_tensors(self):
        for hr, lr in div2k_train:
            assert torch.is_tensor(hr) and torch.is_tensor(lr)

    # Test for checking that all the indices in div2k_test contains two tensors
    def test_test_index_contains_two_tensors(self):
        for hr, lr in div2k_test:
            assert torch.is_tensor(hr) and torch.is_tensor(lr)

    # Test for checking that each index contains a low and high res image in
    # test
    def test_shape_of_images_test(self):
        dataloader = data_module.test_dataloader()
        for index, images in enumerate(dataloader):
            assert len(images) == 2

    # Test for checking that each index contains a low and high res image in
    # train
    def test_shape_of_images_val(self):
        dataloader = data_module.val_dataloader()
        for index, images in enumerate(dataloader):
            assert len(images) == 2
