# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import torchvision
import ssl
import os


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_filepath, output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')

    # Create SSL-context for download
    ssl._create_default_https_context = ssl._create_unverified_context

    # DOWNLOAD HR-IMAGES OF DIV2K
    # Download train set (i.e. image 0-800)
    torchvision.datasets.utils.download_and_extract_archive(
        url='http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip',
        download_root=input_filepath)

    # Delete zip file
    os.remove(input_filepath + '/DIV2K_train_HR.zip')

    # Download validation/test set (i.e. image 800-900)
    torchvision.datasets.utils.download_and_extract_archive(
        url='http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_HR.zip',
        download_root=input_filepath)

    # Delete zip file
    os.remove(input_filepath + '/DIV2K_valid_HR.zip')

    print("Done installing")


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
