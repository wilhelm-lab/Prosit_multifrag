import os
import pandas as pd
import numpy as np
import sys
sys.path.append("/cmnfs/home/j.lapin/projects/shabaz")
from enumerate_tokens import tokenize_modified_sequence as tokenizer

file_path = '/cmnfs/proj/xai/20250312_transformer.parquet'
df = pd.read_parquet(file_path)

# Filter for better predictions
df = df.query("pearson > 0.7")

"""
lines = []
for seq, ce, charge, method in zip(df['modified_sequence'], df['energy'], df['charge'], df['method']):
    tokenseq = tokenizer(seq)
    
    if len(tokenseq) > 30:
        continue
    fullseq = tokenseq + (30-len(tokenseq))*[''] + [str(ce), str(charge), str(method)]
    lines.append(",".join(fullseq))

with open("uvpd_shap_input.csv", 'w') as f:
    f.write("\n".join(lines))
"""
df_ = df[['modified_sequence', 'peptide_length', 'energy', 'charge', 'method']]
df_.index = np.arange(len(df_))
df_.loc[:, 'modified_sequence'] = df_['modified_sequence'].map(lambda x: tokenizer(x))
df_.loc[:, 'full'] = pd.Series(
    df_.loc[:, 'modified_sequence'] +
    df_.loc[:, 'modified_sequence'].map(lambda x: (30-len(x))*['']) + 
    df_.loc[:, 'charge'].map(lambda x: [str(x)]) +
    df_.loc[:, 'energy'].map(lambda x: [str(x)]) + 
    df_.loc[:, 'method'].map(lambda x: [x])
)
df_.to_parquet("shap_input.parquet")
