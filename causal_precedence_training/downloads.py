"""Functions for downloading datafiles on which results depend.
"""

import os
import boto3
import shutil
import tarfile
import tempfile
import causal_precedence_training.locations as locations


def download_signor_pathways():
    """Download SIGNOR pathways
    """
    client = boto3.client('s3')
    with tempfile.TemporaryDirectory() as tmpdir:
        temporary_pathways_location = os.path.join(tmpdir,
                                                   'SIGNOR_pathways.tar.gz')
        client.download_file(locations.S3_BUCKET,
                             f'{locations.S3_DATA_PATH}/SIGNOR_pathways.tar.gz',
                             temporary_pathways_location)
        with tarfile.open(temporary_pathways_location) as tar:
            tar.extractall(path=locations.LOCAL_DATA_HOME)
            # Change folder name to that listed in
            # causal_precedence_training.locations
            shutil.move(os.path.join(locations.LOCAL_DATA_HOME,
                                     'SIGNOR_pathways'),
                        locations.SIGNOR_PATHWAYS)
