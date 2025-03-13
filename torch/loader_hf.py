from datasets import load_dataset
from torch.utils.data import DataLoader
import torch as th
import os
from utils import tokenize_modified_sequence
import re
import sys
sys.path.append("/cmnfs/home/j.lapin/projects/shabaz/data")
from mass_scale import select_ion_dictionary
import yaml
import pandas as pd

def map_fn(
    example, 
    tokenizer, 
    sequence_dictionary, 
    ion_dataframe, 
    max_seq, 
    method_dictionary=None, 
    instrument_dictionary=None
):
    #example['modified_sequence'] = th.tensor(example['modified_sequence'])
    tokenized_sequence = tokenizer(example['modified_sequence'])
    pad = (max_seq - len(tokenized_sequence))*['X']
    tokenized_sequence += pad
    example['intseq'] = th.tensor([sequence_dictionary[x] for x in tokenized_sequence], dtype=th.int32)
    example['charge'] = th.tensor(example['charge'], dtype=th.int32)
    example['ce'] = th.tensor(example['ce'], dtype=th.float32)
    if method_dictionary is not None:
        example['method'] = th.tensor(method_dictionary[example['method']], dtype=th.int32)
    if instrument_dictionary is not None:
        example['instrument'] = th.tensor(instrument_dictionary[example['instrument']], dtype=th.int32)
    intensity = th.zeros(len(ion_dataframe), dtype=th.float32)
    matched = ion_dataframe.loc[example['matched_ions']]
    intensity[matched['index'].tolist()] = th.tensor(example['intensity'], dtype=th.float32) / max(example['intensity'])
    dont_count = ion_dataframe.query(f'length >= {example["pep_len"]} or charge > {example["charge"]}')['index'].tolist()
    intensity[dont_count] = -1
    example['intensity'] = intensity
    return example

def collate_fn(batch_list, full=False):
    use_method = 'method' in batch_list[0]
    use_instrument = 'instrument' in batch_list[0]

    intseq = th.stack([m['intseq'] for m in batch_list])
    charge = th.stack([m['charge'] for m in batch_list])
    ce = th.stack([m['ce'] for m in batch_list])
    if use_method:
        method = th.stack([m['method'] for m in batch_list])
    if use_instrument:
        instrument = th.stack([m['instrument'] for m in batch_list])
    intensity = th.stack([m['intensity'] for m in batch_list])
    
    out = {
        'intseq': intseq,
        'charge': charge,
        'ce': ce,
        'intensity': intensity,
    }
    if use_method:
        out['method'] = method
    if use_instrument:
        out['instrument'] = instrument
    if full:
        out['modified_sequence'] = [m['modified_sequence'] for m in batch_list]
        out['raw_file'] = [m['raw_file'] for m in batch_list]
        out['scan'] = [m['scan'] for m in batch_list]

    return out

class DobjHF:
    def __init__(self, 
        dataset_path: dict,
        tokenizer=None,
        sequence_dictionary_path: str=None,
        ion_counts_path: str=None,
        method_list: list=None,
        instrument_list: list=None,
        top_pks: int=100,
        batch_size: int=100,
        num_workers: int=0,
        **kwargs
    ):
        # Tokenizer
        self.tokenizer = tokenize_modified_sequence if tokenizer is None else tokenizer

        # Input sequence dictionary
        if sequence_dictionary_path is not None:
            self.amod_dic = {
                line.split()[0]:m for m, line in enumerate(open(sequence_dictionary_path))
            }
            self.amod_dic['X'] = len(self.amod_dic)
            self.revdic = {n:m for m,n in self.amod_dic.items()}
        
        if method_list is not None:
            self.method_dic = {method: m for m, method in enumerate(method_list)}
            self.method_dicr = {n:m for m,n in self.method_dic.items()}
        else:
            self.method_dic = None

        if instrument_list is not None:
            self.instrument_dic = {instrument: m for m, instrument in enumerate(instrument_list)}
            self.instrument_dicr = {n:m for m,n in self.instrument_dic.items()}
        else:
            self.instrument_dic = None

        # Output dictionary
        if ion_counts_path is not None:
            try:
                base_directory = "/".join(dataset_path['test'].split('/')[:-1])
                self.ion_df = pd.read_csv(os.path.join(base_directory, "ion_dict.csv"), index_col='full')
            except:
                # FIXME come up with a dynamic way of getting annotation settings
                with open("../data/yaml/annotate.yaml") as  f:
                    annotate = yaml.safe_load(f)
                self.ion_df = select_ion_dictionary(
                    annotate['ion_types'],
                    annotate['max_peptide_length'],
                    annotate['max_product_charge'],
                    ['"p" not in ion', 'counts>100'],
                    counts_path=ion_counts_path,
            )

        # Dataset
        dataset = load_dataset(
            'parquet',
            data_files=dataset_path,
            streaming=True
        )

        # Search for split sizes
        try:
            base_path = "/".join(dataset_path['train'].split('/')[:-1])
            sizes = [int(m) for m in open(os.path.join(base_path, "split_sizes.txt")).read().strip().split("\n")]
            self.sizes = {set_name: size for set_name, size in zip(list(dataset.keys()), sizes)}
        except:
            pass

        # Map to format outputs
        remove_columns = ['score', 'num_peaks', 'pep_len']
        if method_list == None: remove_columns.append("method")
        #if instrument_list == None: remove_columns.append("instrument")
        dataset = dataset.map(
            lambda example: 
            map_fn(
                example,
                tokenizer=self.tokenizer,
                sequence_dictionary=self.amod_dic,
                ion_dataframe=self.ion_df,
                max_seq=kwargs['pep_length'][1],
                method_dictionary=self.method_dic,
                instrument_dictionary=self.instrument_dic,
            ), 
            remove_columns=remove_columns
        )

        # Filter for length
        if 'pep_length' in kwargs.keys():
            dataset = dataset.filter(
                lambda example: 
                (len(example['intseq']) >= kwargs['pep_length'][0]) &
                (len(example['intseq']) <= kwargs['pep_length'][1])
            )
        
        # Filter for charge
        if 'charge' in kwargs.keys():
            dataset = dataset.filter(
                lambda example:
                (example['charge'] >= kwargs['charge'][0]) &
                (example['charge'] <= kwargs['charge'][1])
            )

        # Shuffle the dataset
        if 'buffer_size' in kwargs.keys():
            dataset['train'] = dataset['train'].shuffle(buffer_size=kwargs['buffer_size'])
        else:
            dataset['train'] = dataset['train'].shuffle()
        
        self.dataset = dataset

        # Dataloaders
        num_workers = min(self.dataset['train'].n_shards, num_workers)
        self.dataloader = {
            'train': self.build_dataloader(dataset['train'], batch_size, num_workers, full=False),
            'val':   self.build_dataloader(dataset['val']  , batch_size, 0, full=False),
            'test':  self.build_dataloader(dataset['test'] , batch_size, 0, full=True),
        }

    def build_dataloader(self, dataset, batch_size, num_workers, full=False):
        return DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            collate_fn=lambda x: collate_fn(x, full)
        )

    def save_ion_dictionary(self, path):
        self.ion_df.to_csv(path, index=True)

if __name__ == "__main__":

    dataset_path = {
        'train': "/cmnfs/proj/latent_model/data/intensity/train/*parquet",
        'val': "/cmnfs/proj/latent_model/data/intensity/val/*parquet",
    }
    
    batch_size = 100
    loader = DobjHF(
        dataset_path=dataset_path,
        tokenizer=tokenize_modified_sequence,
        dictionary_path='ns_dictionary.txt',
        pep_length=[6,30],
        charge=[1,6],
        batch_size=batch_size,
        buffer_size=10000,
        num_workers=8,
    )
    
    from time import time
    for m, batch in enumerate(loader.dataloader['train']):
        print("\r%d"%m, end='')
        if m == 10:
            start = time()
        if m == 1010:
            end = time()
            break
        
    print()
    spectra_per_second = 1000 * batch_size / (end-start)
    print("Spectra / second = %.1f"%spectra_per_second)

