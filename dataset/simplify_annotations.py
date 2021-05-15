import os

import pandas as pd

def main(input_file: str, output_file: str) -> None:
    df_out = pd.DataFrame(columns=['ID', 'sentiments'])
    df = pd.read_csv(input_file, sep='\t')
    interesting_columns = ['ID']
    for col in df.columns:
        if 'ImageOnly' in col:
            sentiment = col.split(':')[1].strip()
            df = df.rename(columns={col:sentiment})
            interesting_columns.append(sentiment)
    df = df[interesting_columns]

    for i, row in df.iterrows():
        row_dict = row.to_dict()
        idx = row_dict.pop('ID')
        new_row = {'ID':idx, 'sentiments':str(row_dict)}
        df_out = df_out.append(new_row, ignore_index=True)

    df_out.to_csv(output_file, sep=';', index=False)

if __name__ == '__main__':
    main(os.path.join('input', 'WikiArt-Emotions-All.tsv'), os.path.join('output', 'annotations.csv'))