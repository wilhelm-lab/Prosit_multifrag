import re
import os
import datetime
import torch as th

def timestamp():
    dt = str(datetime.datetime.now()).split()
    dt[-1] = re.sub(':', '-', dt[-1]).split('.')[0]
    return "_".join(dt)

def create_experiment(directory, svwts=False):
    os.mkdir(directory)
    os.mkdir('%s/yaml'%directory)
    os.system("cp ./yaml/*.yaml %s/yaml/"%directory)
    if svwts: 
        os.mkdir('%s/weights'%directory)

def message_board(line, path):
    with open(path, 'a') as F:
        F.write(line)

def save_full_model(model, optimizer, svdir, ext=''):
    if model is not None:
        th.save(
            model.state_dict(),
            "%s/weights/model_%s"%(svdir, ext)
        )
    if optimizer is not None:
        save_optimizer_state(
            optimizer, '%s/weights/opt_last'%svdir
        )

def save_optimizer_state(opt, fn):
    th.save(opt.state_dict(), fn)

def load_optimizer_state(opt, fn, device):
    opt.load_state_dict(th.load(fn, map_location=device))

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

    return tokenized

def Dict2dev(Dict, device, inplace=False):
    if inplace:
        for key in Dict.keys():
            if type(Dict[key]) not in [list]:
                Dict[key] = Dict[key].to(device)
        return True
    else:
        return {a: b.to(device) for a,b in Dict.items() if type(b)!=list}

def global_grad_norm(model):
    return sum([m.grad.detach().square().sum().item() for m in model.parameters() if m.requires_grad])**0.5
