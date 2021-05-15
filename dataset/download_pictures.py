import os
from urllib import request
import ssl

import pandas as pd

def main(input_file: str, output_dir: str) -> None:
    ssl._create_default_https_context = ssl._create_unverified_context
    df = pd.read_csv(input_file, sep='\t')
    for i, row in df.iterrows():
        output_file = os.path.join(output_dir, '{}.jpg'.format(row['ID']))
        if not os.path.isfile(output_file):
            request.urlretrieve(row['Image URL'].replace('use2-', ''), output_file)

if __name__ == '__main__':
    main(os.path.join('input', 'WikiArt-Emotions-All.tsv'), os.path.join('output', 'pictures'))