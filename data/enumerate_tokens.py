import re
import pandas as pd
import os
import glob
from collections import Counter, OrderedDict
import sys

def tokenize_modified_sequence(modseq):
    tokenized = []
    modseq = re.sub('-|(\[])', '', modseq) # remove - or []
    #modseq = re.sub('(\[]-)|(\-\[])','',modseq)
    
    pos = 0
    while pos < len(modseq):
        character = modseq[pos]
        hx = ord(character)
        if character == '[':
            ahead = 1
            mod = []
            while character != ']':
                mod.append(character)
                character = modseq[pos+ahead]
                ahead += 1
            token = "".join(mod) + ']'
            if pos != 0:
                tokenized[-1] += token
            else:
                tokenized.append(token)
            pos += ahead - 1
        else:
            tokenized.append(character)
        pos += 1
    
    # N terminal
    #if tokenized[0][0] == '[':
    #    tokenized[1] = "".join([tokenized[1],tokenized[0]])
    #    tokenized.pop(0)

    return tokenized

if __name__ == "__main__":

    directory = f"/cmnfs/data/proteomics/shabaz_exotic/processed/merged_search/{sys.argv[1]}"
    list_of_files = glob.glob(os.path.join(directory, "*parquet"))
    #list_of_files += glob.glob(os.path.join(directory, 'val/*parquet'))

    tokens = {}
    for file in list_of_files:
        print(file)
        df = pd.read_parquet(file, columns=['MODIFIED_SEQUENCE'])

        for m, modseq in enumerate(df['MODIFIED_SEQUENCE']):
            print("\r%d/%d"%(m, len(df)), end='')
            tokenized = tokenize_modified_sequence(modseq)
            dic = Counter(tokenized)
            for token, count in dic.items():
                if token not in tokens:
                    tokens[token] = 0
                tokens[token] += count
        print()


    od = OrderedDict(sorted(tokens.items()))
    with open(os.path.join(directory, "token_dictionary.txt"), 'w') as f:
        for a,b in od.items():
            f.write("%s %d\n"%(a,b))
      
