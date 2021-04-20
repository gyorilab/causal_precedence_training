import pystow

# S3 bucket for raw data
S3_BUCKET = 'bigmech'
# Path to data within bucket
S3_DATA_PATH = 'causal_precedence'
# Path to where data is stored locally
LOCAL_DATA_HOME = pystow.join('causal_precedence_training')
# Path to SIGNOR pathways
SIGNOR_PATHWAYS_DIRECTORY = pystow.join('causal_precedence_training',
                                        'SIGNOR_pathways')
# Path to triples used as input to generate causal precedence datasets
TRIPLES_DIRECTORY = pystow.join('causal_precedence_training',
                                'causal_triples')
