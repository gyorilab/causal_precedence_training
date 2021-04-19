import os
import pystow

# S3 bucket for raw data
S3_BUCKET = 'bigmech'
# Path to data within bucket
S3_DATA_PATH = 'causal_precedence'
# Path to where data is stored locally
LOCAL_DATA_HOME = pystow.join('causal_precedence_training')
SIGNOR_PATHWAYS = os.path.join(LOCAL_DATA_HOME, 'Signor_pathways')

