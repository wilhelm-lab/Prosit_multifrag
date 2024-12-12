import pandas as pd
import sys
sys.path.append("/cmnfs/home/j.lapin/projects/shabaz")
from mass_scale import tokenize_modified_sequence as tokenizer, select_ion_dictionary, Scale
import numpy as np
from glob import glob
from tqdm import tqdm
import yaml
import os
scale = Scale()

def calc_mz(row):
    modified_sequence = row['MODIFIED_SEQUENCE']
    charge = row['PRECURSOR_CHARGE']
    return scale.calcmass(modified_sequence, charge, 'p')

# only works for NCE=30
def calculate_ev(row):
    mz = row['mz']
    charge = row['PRECURSOR_CHARGE']
    if charge == 2:
        ev = 0.0347811732677003*mz + 3.90532075915273
    elif charge == 3:
        ev = 0.0329095703871935*mz + 3.86650294096255
    elif charge == 4:
        ev = 0.0309364348610361*mz + 3.91491149272491
    elif charge == 5:
        ev = 0.0291682416131185*mz + 3.8149680577527
    elif charge == 6:
        ev = 0.0291682416131185*mz + 3.8149680577527
    elif charge == 7:
        ev = 30
    else:
        ev = 30

    return ev

with open("yaml/annotate.yaml") as f:
    annotate = yaml.safe_load(f)
with open("yaml/create_dataset.yaml") as f:
    config = yaml.safe_load(f)

# Make sure the destination directory exists
svdir = "allinone" if len(config['method']) > 1 else config['method'][0]
if not os.path.exists(f"/cmnfs/data/proteomics/shabaz_exotic/processed/parquet/{svdir}"):
    os.makedirs(f"/cmnfs/data/proteomics/shabaz_exotic/processed/parquet/{svdir}")
if not os.path.exists(f"/cmnfs/data/proteomics/shabaz_exotic/processed/merged_search/{svdir}"):
    os.makedirs(f"/cmnfs/data/proteomics/shabaz_exotic/processed/merged_search/{svdir}")

#############
# Load psms #
#############

rename = {
    'RAW_FILE': 'raw_file',
    'SCAN_NUMBER': 'scan',
    'MODIFIED_SEQUENCE': 'modified_sequence',
    'PRECURSOR_CHARGE': 'charge',
    'COLLISION_ENERGY': 'ce',
    'PEPTIDE_LENGTH': 'pep_len',
    'MASS': 'mass',
    'SCORE': 'score',
    'REVERSE': 'reverse',
    'INTENSITIES': 'intensity',
    'num_ann_peaks': 'num_peaks',
    'matched_inds': 'matched_inds',
    'matched_ions': 'matched_ions',
}

df = pd.DataFrame()
for frag_directory in config['method']:
    parquet_files = glob(f"/cmnfs/data/proteomics/shabaz_exotic/processed/merged_search/{frag_directory}/all_psms*parquet")
    for file in parquet_files:
        df_ = pd.read_parquet(file, columns=list(rename.keys()))
        df_['method'] = pd.Series(df_.shape[0]*[frag_directory])

        # If index file exisits, use it
        split = file.split('/')[:-1]
        split[-2] = "parquet"
        filtered_index_path = "/".join(split + ["filtered_index.txt"])
        if os.path.exists(filtered_index_path):
            loc_indices = np.loadtxt(filtered_index_path).astype(int)
            df_ = df_.loc[loc_indices]

        if frag_directory == "HCD":
            #mz1 = df_.apply(calc_mz, axis=1)
            # Quick and dirty method
            df_['mz'] = df_['MASS'] / df_['PRECURSOR_CHARGE'] + 1.0043171043797656
            df_['COLLISION_ENERGY'] = df_.apply(calculate_ev, axis=1)
            del df_['mz']
        else:
            df_['COLLISION_ENERGY'] = pd.Series(df_.shape[0]*[0.])
            df_['COLLISION_ENERGY'].fillna(0, inplace=True)
        del df_['MASS']
        
        df = pd.concat([df, df_])

df = df.rename(columns=rename)
df.index = np.arange(len(df))

# Collect all ion_counts and tokens, if allinone
if svdir == 'allinone':
    all_ions = {}
    all_tokens = {}
    for frag_directory in config['method']:
        ion_counts_file = glob(f"/cmnfs/data/proteomics/shabaz_exotic/processed/merged_search/{frag_directory}/ion_counts.tab")[0]
        token_counts_file = glob(f"/cmnfs/data/proteomics/shabaz_exotic/processed/merged_search/{frag_directory}/token_dictionary.txt")[0]

        ion_counts = {line.split()[0]: int(line.split()[1]) for line in open(ion_counts_file).read().strip().split("\n")}
        for ion, count in ion_counts.items():
            if ion not in all_ions:
                all_ions[ion] = 0
            all_ions[ion] += count

        token_counts = {line.split()[0]: int(line.split()[1]) for line in open(token_counts_file).read().strip().split("\n")}
        for token, count in token_counts.items():
            if token not in all_tokens:
                all_tokens[token] = 0
            all_tokens[token] += count

    with open(f"/cmnfs/data/proteomics/shabaz_exotic/processed/merged_search/{svdir}/ion_counts.tab", "w") as f:
        lines = ["%s\t%d"%(ion, count) for ion, count in all_ions.items()]
        f.write("\n".join(lines))

    with open(f"/cmnfs/data/proteomics/shabaz_exotic/processed/merged_search/{svdir}/token_dictionary.txt", "w") as f:
        lines = ["%s %d"%(token, count) for token, count in all_tokens.items()]
        f.write("\n".join(lines))

#############
# Filtering #
#############

# Remove decoys
df = df.query("reverse == False")
df = df.drop(["reverse"], axis=1)

# Only matching intensities
# Do I want to include all matched ions, or only those that I plan to be in my dictionary?
iondict = select_ion_dictionary(
    annotate['ion_types'],
    annotate['max_peptide_length'],
    annotate['max_product_charge'],
    config['criteria'],
    config['counts_path'],
)
masks = df['matched_ions'].map(lambda x: [m in iondict.index for m in x])
df['intensity'] = pd.Series([a[b][c] for a,b,c in zip(df['intensity'], df['matched_inds'], masks)], index=df.index)
df['matched_ions'] = pd.Series([a[b] for  a,b in zip(df['matched_ions'], masks)], index=df.index)
df['num_peaks'] = pd.Series(np.vectorize(sum)(masks), index=df.index)
del df['matched_inds']

# Remove spectra that had no annotations
df = df.query(f"num_peaks >= {config['peaks_min']}")

# Remove bad spectra based on spectral entropy
df = df.assign(entropy=pd.Series(df['intensity'].map(lambda x: -sum((x/max(x))*np.log((x/max(x))))).values))

############################
# Remove duplicate spectra #
############################

# Every unique combination of sequence and charge
all_unique_seqch, counts = np.unique(['%s_%d_%s'%m for m in zip(df['modified_sequence'], df['charge'], df['method'])], return_counts=True)
all_unique_seqch = all_unique_seqch[counts>1]
all_drop_indices = np.array([], dtype=np.int64)
for i, uniq in enumerate(pbar := tqdm(all_unique_seqch)):
    pbar.set_description(f"len(df)={len(df)}")
    sequence, charge, method = uniq.split('_')

    query = "modified_sequence == '%s' and charge == %s and method == '%s'"%(sequence, charge, method)
    df_query = df.query(query)
    
    # If there is more than one match, choose the highest score
    assert len(df_query)>1, query
    drop_indices = df_query['score'].argsort()[:-1].index.values
    all_drop_indices = np.append(all_drop_indices, drop_indices)
    
    # Drop indices at regular intervals
    if len(all_drop_indices) > 1000:
        df = df.drop(all_drop_indices)
        all_drop_indices = np.array([], dtype=np.int64)
df = df.drop(all_drop_indices)

# save the index vector
np.savetxt(f"/cmnfs/data/proteomics/shabaz_exotic/processed/parquet/{svdir}/filtered_index.txt", df.index.to_numpy(), fmt='%d')

# This will allow one to map test results back to their original spectrum
df['merged_index'] = df.index.tolist()

########################
# Output parquet files #
########################

# Cast data types
df = df.astype({
    'charge': 'int32', 
    'ce': 'float32', 
    'pep_len': 'int32', 
    'score': 'float32', 
    'num_peaks': 'int32',
    'merged_index': 'int32',
})

# Shuffle the dataset
df = df.sample(frac=1)

# Split
splits = [0.8, 0.1, 0.1]
splits = [int(m*len(df)) for m in splits]
splits = np.cumsum(splits)
splits[-1] = len(df)
split_sizes = np.append(splits[0], splits[1:] - splits[:-1])
np.savetxt(f"/cmnfs/data/proteomics/shabaz_exotic/processed/parquet/{svdir}/split_sizes.txt", split_sizes, fmt='%d')

# Shard training set
n_shards = config['n_shards']
sub_splits = np.cumsum([0] + n_shards * [splits[0] // n_shards])
sub_splits[-1] = splits[0]
for i in range(n_shards):
    df.iloc[sub_splits[i] : sub_splits[i+1]].to_parquet(f"/cmnfs/data/proteomics/shabaz_exotic/processed/parquet/{svdir}/train{i}.parquet")
df.iloc[splits[0] : splits[1]].to_parquet(f"/cmnfs/data/proteomics/shabaz_exotic/processed/parquet/{svdir}/val.parquet")
df.iloc[splits[1] : ].to_parquet(f"/cmnfs/data/proteomics/shabaz_exotic/processed/parquet/{svdir}/test.parquet")

# Save the ion dictionary along with the training data, since the dataset contains filtered ions
iondict.to_csv(f"/cmnfs/data/proteomics/shabaz_exotic/processed/parquet/{svdir}/ion_dict.csv", index=True)
