"""Functions for downloading datafiles
"""

import os
import boto3
import shutil
import tarfile
import causal_precedence_training.locations as locations


def download_signor_pathways():
    """Download SIGNOR pathways into data directory
    """
    client = boto3.client('s3')
    pathways_tarball_location = os.path.join(locations.LOCAL_DATA_HOME,
                                             'SIGNOR_pathways.tar.gz')
    client.download_file(locations.S3_BUCKET,
                         f'{locations.S3_DATA_PATH}/SIGNOR_pathways.tar.gz',
                         pathways_tarball_location)
    with tarfile.open(pathways_tarball_location) as tar:
        tar.extractall(path=locations.LOCAL_DATA_HOME)
    os.remove(pathways_tarball_location)
    # Change folder name to that listed in
    # causal_precedence_training.locations
    shutil.move(os.path.join(locations.LOCAL_DATA_HOME,
                             'SIGNOR_pathways'),
                locations.SIGNOR_PATHWAYS)
