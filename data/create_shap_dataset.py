import os
import pandas as pd
import sys
sys.path.append("/cmnfs/home/j.lapin/projects/shabaz")
from enumerate_tokens import tokenize_modified_sequence as tokenizer

file_path = '/cmnfs/home/j.lapin/projects/shabaz/torch/save/test_results_241130_UVPD.parquet'
df = pd.read_parquet(file_path)

# Filter for better predictions
df = df.query("pearson > 0.7")

lines = []
for seq, ce, charge in zip(df['modified_sequence'], df['energy'], df['charge']):
    tokenseq = tokenizer(seq)
    
    if len(tokenseq) > 30:
        continue
    fullseq = tokenseq + (30-len(tokenseq))*[''] + [str(ce), str(charge)]
    lines.append(",".join(fullseq))

with open("uvpd_shap_input.csv", 'w') as f:
    f.write("\n".join(lines))
