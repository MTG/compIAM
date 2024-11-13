"""Utilities for downloading from the web.
PART OF THE CODE IS TAKEN FROM mir-dataset-loaders/mirdata. Kudos to all authors :)
"""

import os
import zipfile
import hashlib
import requests

from tqdm import tqdm
from smart_open import open

from compiam.utils import get_logger

logger = get_logger(__name__)


def md5(file_path):
    """Get md5 hash of a file.

    Args:
        file_path (str): File path

    Returns:
        str: md5 hash of data in file_path

    """
    hash_md5 = hashlib.md5()
    with open(file_path, "rb", compression="disable") as fhandle:
        for chunk in iter(lambda: fhandle.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def download_remote_model(
    download_link, download_checksum, download_path, force_overwrite=False
):
    """Elegantly download model from Zenodo.

    IMPORTANT DISCLAIMER: Part of the code is taken from mir-dataset-loders/mirdata :)

    :param download_link: link to remote model to download
    :param download_checksum: checksum of the downloaded model
    :param download_path: path to save the downloaded model
    :param force_overwrite: if True, overwrite existing file
    """
    if "zenodo.org" not in download_link:
        raise ValueError("Only Zenodo download link are supported.")
    if len(os.listdir(download_path)) > 0 and not force_overwrite:
        logger.warning(
            f"""Files already exist at {download_path}. Skipping download.
            Please make sure these are correct.
            Otherwise, run the .download_model() method with force_overwrite=True.
        """
        )
        return
    else:
        local_filename = download_zip(download_link, download_path)
        # Check the checksum
        checksum = md5(local_filename)
        if download_checksum != checksum:
            raise IOError(
                "{} has an MD5 checksum ({}) "
                "differing from expected ({}), "
                "file may be corrupted.".format(
                    download_path, checksum, download_checksum
                )
            )
        # Unzip it
        extract_zip(local_filename, download_path, cleanup=True)
        logger.info("Files downloaded and extracted successfully.")
        return


def download_zip(url, root_path):
    """Download a ZIP file from a URL."""
    # Get the file name from the URL
    local_filename = os.path.join(
        root_path,
        url.split("/")[-1].split("?")[0]
    )

    # Stream the download and save the file
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        total_size = int(r.headers.get("content-length", 0))
        chunk_size = 8192
        with open(local_filename, "wb") as f, tqdm(
            total=total_size, unit="iB", unit_scale=True
        ) as pbar:
            for chunk in r.iter_content(chunk_size=chunk_size):
                if chunk:  # filter out keep-alive new chunks
                    f.write(chunk)
                    pbar.update(len(chunk))
    logger.info(f"Download complete: {local_filename}")
    return local_filename


def extract_zip(local_filename, extract_to=".", cleanup=True):
    """Extract a ZIP file into a given folder."""
    # Check if it's a zip file
    if zipfile.is_zipfile(local_filename):
        logger.info(f"Extracting {local_filename}...")
        with zipfile.ZipFile(local_filename, "r") as zip_ref:
            zip_ref.extractall(extract_to)
        logger.info(f"Extraction complete: Files extracted to {extract_to}")
    else:
        logger.info(f"{local_filename} is not a valid ZIP file.")
    if cleanup:
        os.remove(local_filename)
