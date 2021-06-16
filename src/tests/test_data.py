import torch

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
        for index, images in div2k_train:
            assert len(images) == 2

    # Test for checking that each index contains a low and high res image in test
    def test_shape_of_images(self):
        for index, images in div2k_test:
             assert len(images) == 2

    # Test that each label is reresented once
    def test_occurance_of_labels_train(self):
        label_set = set()
        test_labels = list(range(0, 10))
        for images, labels in train_dataset:
            label_set.add(labels.item())

        assert test_labels == list(label_set)
