"""Functions for downloading datafiles."""

import os
import shutil
import tarfile

import boto3
from causal_precedence_training import locations


def download_signor_pathways():
    """Download SIGNOR pathways into data directory."""
    client = boto3.client('s3')
    pathways_tarball_location = os.path.join(locations.LOCAL_DATA_HOME,
                                             'SIGNOR_pathways.tar.gz')
    client.download_file(locations.S3_BUCKET,
                         f'{locations.S3_DATA_PATH}/SIGNOR_pathways.tar.gz',
                         pathways_tarball_location)
    with tarfile.open(pathways_tarball_location) as tar:
        def is_within_directory(directory, target):
            
            abs_directory = os.path.abspath(directory)
            abs_target = os.path.abspath(target)
        
            prefix = os.path.commonprefix([abs_directory, abs_target])
            
            return prefix == abs_directory
        
        def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
        
            for member in tar.getmembers():
                member_path = os.path.join(path, member.name)
                if not is_within_directory(path, member_path):
                    raise Exception("Attempted Path Traversal in Tar File")
        
            tar.extractall(path, members, numeric_owner=numeric_owner) 
            
        
        safe_extract(tar, path=locations.LOCAL_DATA_HOME)
    os.remove(pathways_tarball_location)
    # Change folder name to that listed in
    # causal_precedence_training.locations (defensive programming)
    shutil.move(os.path.join(locations.LOCAL_DATA_HOME,
                             'SIGNOR_pathways'),
                locations.SIGNOR_PATHWAYS_DIRECTORY)
