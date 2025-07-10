import argparse

from protein_classification.data.cellatlas import get_cellatlas_filepaths_and_labels


parser = argparse.ArgumentParser()
parser.add_argument("--ckpt_dir", type=str, required=True)