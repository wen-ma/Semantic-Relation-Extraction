import os
import glob

import pandas as pd
import numpy as np

"""
Find all the English files in the directory and concat as a df
    - remove unnecessary infos
    - drop columns with NaN values
    - reset the index
    - split on the delimiter '\t' 
    - drop the 5th column without values
"""
all_en_files = glob.glob(os.path.join('../data/curation', '*_en.tsv'))

df = pd.concat((pd.read_csv(file, header=None, sep='\n') for file in all_en_files), ignore_index=True)
df = df[df[0].str.contains('# webanno') == False]
df = df[df[0].str.contains('#id=') == False]
df = df[df[0].str.contains('#text=') == False]
df = df.dropna(axis=1, how='all')
df = df.reset_index(drop=True)

df = df[0].str.split('\t', expand=True)
df = df.drop([5], axis=1)

"""
Deal with the column 0: token_id
    - 1. keep only the order of the tokens, e.g. make 1-12 to 12
    - 2. convert dtype 'object' to int
    - 3. set row id starts from 0
"""
df[0] = df[0].apply(lambda x: x.split('-')[-1])
df[0] = df[0].apply(lambda x: int(x))
df[0] = df[0].apply(lambda x: x - 1)

"""
Deal with the column 2: BIO
    - 1. to tackle the overlap problem, the multiple BIO tags should be first split into multiple columns
        - add prefix and fill None values with 'O'
    - 2. merge df and df_BIO
"""
df_BIO = df[2].str.split('|', expand=True).add_prefix('BIO').fillna(value='O')
df = df.join(df_BIO)

"""
Deal with the column 3: relation
    - 1. replace _ with N
    - 2. split the string values into a list
"""
df[3] = df[3].replace('_', 'N')
df[3] = df[3].apply(lambda x: x.split('|'))

"""
Deal with the column 4: head
    - 1. replace _ with np.NaN
    - 2. split the str on separators
        - e.g, '2-103|2-103|2-240' --> ['2', '103','2', '103','2', '240']
    - 3. keep the values on the even positions if the values type is a list
        - e.g, ['2', '103','2', '103','2', '240'] --> ['103','103','240']
        - list[::2] --> odd positions | list[1::2] --> even positions
    - 4. convert data type from str to int in a list
        - e.g, ['103','103','240'] --> [103, 103, 240]
    - 5. to align the pos of token_id, each element minus 1 in a list
        - e.g, [103, 103, 240] --> [102, 102, 239] because the orignal original token_id starts from 1
    - 6. fill NaN with values from column 0: token_id
    - 7. add [] to the single int values
"""
df[4] = df[4].replace('_', np.NaN)
df[4] = df[4].str.split('[||-]', expand=False)
df[4] = df[4].apply(lambda x: x[1::2] if type(x) == list else x)
df[4] = df[4].apply(lambda x: list(map(int, x)) if type(x) == list else x)
df[4] = df[4].apply(lambda x: list(map(lambda i: i - 1, x)) if type(x) == list else x)
df[4].fillna(df[0], inplace=True)
df[4] = df[4].apply(lambda x: '[' + str(x) + ']' if type(x) == int else x)

"""
Rename df columns with the corresponding names
"""
new_col_names = ['token_id', 'token', 'BIO', 'relation', 'head']
old_col_names = df.columns[[0, 1, 2, 3, 4]]
df.rename(columns=dict(zip(old_col_names, new_col_names)), inplace=True)

"""
Add doc ID to identify a certain doc for parsing data later
    - filter the indic where the token_id value == True, which is the beginning pos of a document and make it as a list
    - align the pos in the whole df
    - iterate over the df and add the new rows
"""
zero_idx = df.loc[df['token_id'] == 0].index.tolist()
shift_idx = list(map(lambda x: x+zero_idx.index(x), zero_idx))

for idx in shift_idx:
    new_row = pd.DataFrame({'token_id': '#doc' + ' ' + str(shift_idx.index(idx) + 1)}, index=[idx])
    df.index = df.index + 0.5
    df = df.append(new_row)
    df = df.sort_index().reset_index(drop=True)

# df.to_csv('./data/semreldata.tsv', sep='\t', header=False, index=False)


