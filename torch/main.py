import torch as th
import yaml
from loader_hf import DobjHF
from utils import tokenize_modified_sequence
from models.peptide_encoder import PeptideEncoder
from losses import masked_spectral_distance
import utils as U
from collections import deque
from time import time
import sys
import numpy as np
import wandb
import os
import pandas as pd
device = th.device("cuda" if th.cuda.is_available() else 'cpu')

with open("yaml/master.yaml", "r") as f:
    master_config = yaml.safe_load(f)
with open("yaml/loader.yaml", "r") as f:
    load_config = yaml.safe_load(f)
with open("yaml/model.yaml", "r") as f:
    model_config = yaml.safe_load(f)
if load_config['method_list'] is not None:
    model_config['num_methods'] = len(load_config['method_list'])

##################################################
#                   Dataloader                   #
##################################################

load_config['tokenizer'] = tokenize_modified_sequence if load_config['tokenizer']=='mine' else None
load_config['batch_size'] = master_config['batch_size'] # OVERRIDE
dobj = DobjHF(**load_config)

##################################################
#                     Model                      #
##################################################

model = PeptideEncoder(
    tokens = len(dobj.amod_dic),
    final_units = len(dobj.ion_df),
    max_charge = load_config['charge'][-1],
	**model_config
)
model.to(device)
total_parameters = sum([m.numel() for m in model.parameters() if m.requires_grad])
print(f"Total encoder parameters: {total_parameters:,}")

starting_lr = master_config['lr'] if master_config["warmup_steps"]==0 else 1e-7
opt = th.optim.Adam(model.parameters(), master_config['lr'])

###################################################
#                     WandB                       #
###################################################
if (len(sys.argv)==1) and master_config['log_wandb']:
    wandb.init(
        project=master_config['project'],
        entity='joellapin',
        config={
            'master': master_config,
            'model': model_config,
            'loader': load_config,
            'Total parameters': total_parameters,
        },
    )

##################################################
#                Loss function                   #
##################################################

from losses import cosine_score, masked_spectral_distance
from losses import masked_pearson_correlation_distance
all_losses = {
    'cosine_score': cosine_score, 
    'masked_spectral_distance': masked_spectral_distance
}
assert master_config['loss_function'] in all_losses.keys(), (
    f"{master_config['loss_function']} not available"
)
loss_function = all_losses[master_config['loss_function']]
all_evals = {
    'pearson': masked_pearson_correlation_distance
}
assert master_config['eval_function'] in all_evals.keys(), (
    f"{master_config['eval_function']} not available"
)
eval_function = all_evals[master_config['eval_function']]

##################################################
#                  Train step                    #
##################################################

def train_step(batch, opt):
    # Verify training mode, device, and zero grads
    model.train()
    opt.zero_grad()

    batch = U.Dict2dev(batch, device, inplace=False)

    inp = {
        'intseq': batch['intseq'],
        'charge': batch['charge'],
        'energy': batch['ce'],
        'method': batch['method'] if 'method' in batch else None,
    }
    prediction = model(**inp)
    
    loss = loss_function(y_true=batch['intensity'], y_pred=prediction)
    loss = loss.mean()
    
    loss.backward()
    opt.step()

    model.global_step +=1

    return loss

###################################################
#             Training and Evaluation             #
###################################################

def evaluation(dset='val', save_df=False):
    model.eval()
    
    df = pd.DataFrame()
    
    sum_instances = 0
    sum_loss = 0
    eval_score_ = 0
    total_eval_steps = dobj.sizes[dset] // master_config['batch_size']
    for step, batch in enumerate(dobj.dataloader[dset]):
        print("\rEvaluation batch %d/%d"%(step, total_eval_steps), end=50*" ")
        
        sum_instances += len(batch['charge'])
        batchdev = U.Dict2dev(batch, device, inplace=False)

        inp = {
            'intseq': batchdev['intseq'],
            'charge': batchdev['charge'],
            'energy': batchdev['ce'],
            'method': batchdev['method'] if 'method' in batch else None,
        }
        with th.no_grad():
            prediction = model(**inp)

        loss = loss_function(y_true=batchdev['intensity'], y_pred=prediction)
        sum_loss += loss.sum().item()
        eval_score = eval_function(batchdev['intensity'], prediction)
        eval_score_ += eval_score.sum().item()

        if save_df:
            #C = list(map(lambda x: "".join([dobj.revdic[m] for m in x if m!=dobj.amod_dic['X']]), batch['intseq'].cpu().tolist()))
            A = {
                'raw_file': batch['raw_file'],
                'scan': batch['scan'],
                'modified_sequence': batch['modified_sequence'],
                'charge': batch['charge'].cpu().tolist(),
                'energy': batch['ce'].cpu().tolist(),
                master_config['loss_function']: loss.cpu().tolist(),
                'pearson': (1-eval_score).cpu().tolist(),
                'peptide_length': (batch['intseq']!=23).sum(1).cpu().tolist(),
                'num_peaks': (batch['intensity']>0).sum(1).cpu().tolist(),
                'predicted_intensities': (prediction/prediction.max(1, keepdims=True)[0]).cpu().tolist(),
                'true_intensities': batch['intensity'].cpu().tolist(),
            }
            if 'method' in batch: A['method']=[dobj.method_dicr[int(i)] for i in batch['method']]
            df = pd.concat([df, pd.DataFrame(A)])

    return {
        'train_loss': sum_loss / sum_instances,
        'eval_score': eval_score_ / sum_instances,
        'dataframe': df,
    }

def train(epochs=1, runlen=50, svfreq=3600):
    
    # Shorthand
    bs = master_config['batch_size']
    msg = master_config['log']
    swt = master_config['svwts']
    
    # Create experiment directory in save/
    if (msg or swt):
        timestamp = U.timestamp()
        svdir = 'save/' + timestamp
        U.create_experiment(svdir, svwts=master_config['svwts'])
        if master_config['svwts']: 
            U.save_full_model(model, opt, svdir, ext='last')
            dobj.ion_df.to_csv(os.path.join(svdir, "filtered_ion_dict.csv"), index=True)
    else:
        svdir = './' # for establishing ds objects below
    
    # Log starting messages and start collection all lines
    if msg:
        line = f"Experiment header: {master_config['header']}\nTotal parameters: {total_parameters():,}\n"
        U.message_board(line, "%s/epochout.txt"%svdir)
        line = "%s\n%s\n"%(timestamp, master_config['header'])
        allepochlines = [line]

    # Warmup lr
    warmup = np.linspace(1e-7, master_config['lr'], master_config['warmup_steps'])
    
    # First eval
    evals = evaluation()
    top_eval_loss = evals['train_loss']
    if master_config['log_wandb']: wandb.log({'Validation loss': evals['eval_score']})

    # Train
    running_time = deque(maxlen=runlen) # Full time
    load_time = deque(maxlen=runlen) # load_batch time
    graph_time = deque(maxlen=runlen) # train_step time
    running_loss = deque(maxlen=runlen)
    svtime = time()
    sys.stdout.write("\rStarting training for %d epochs\n"%epochs)
    
    loss_list = []
    max_steps_tick=False
    for epoch in range(epochs):
        start_epoch = time()
        
        dobj.dataset['train'].set_epoch(epoch)
        total_train_steps = dobj.sizes['train'] // master_config['batch_size']
        start_load = time()
        for step, batch in enumerate(dobj.dataloader['train']):
            running_time.append(0 if step==0 else time()-start_step)
            start_step = time()
            load_time.append(start_step-start_load)
            
            # Warmup lr
            if model.global_step < master_config['warmup_steps']:
                opt.param_groups[0]["lr"] = warmup[model.global_step]
            else:
                opt.param_groups[0]["lr"] = master_config['lr']

            # Train model for a step
            TT=time()
            loss = train_step(batch, opt)
            
            # Save running stats
            loss = loss.detach().cpu().numpy()
            running_loss.append(loss)
            running_time.append(time()-start_step)
            graph_time.append(time()-TT)
                        
            # Stdout
            mean_loss = np.mean(running_loss)
            if step%10==0:
                loss_string = "%.4f"%mean_loss
                sys.stdout.write(
                    "\r\033[KStep %6d/%6d, loss=%s (%.3f,%.3f,%.3f s)"%(
                        step, total_train_steps, loss_string, 
                        np.mean(running_time), np.mean(load_time), 
                        np.mean(graph_time)
                    )
                )

            # WandB
            if master_config['log_wandb']:
                wandb.log({
                    'Global_step': model.global_step.item(),
                    'Epoch': epoch,
                    'Train loss': loss, 
                    'Running train loss': mean_loss,
                    'Learning rate': opt.param_groups[0]["lr"],
                })
                global_grad_norm = U.global_grad_norm(model)
                wandb.log({'Global gradient norm': global_grad_norm})
            
            # Saving weights and testing
            if time()-svtime > svfreq:
                if swt:
                    U.save_full_model(model, opt, svdir, ext='last')
                svtime = time()
            
            start_load = time()

        # End of epoch
        if max_steps_tick:
            break
        
        # Evaluation
        evals = evaluation()
        Line = "Validation loss at epoch %d: %.6f\n"%(epoch, evals['eval_score'])
        if master_config['log_wandb']: wandb.log({'Validation loss': evals['eval_score'], 'Epoch': epoch})
        if swt:
            weight_files = os.listdir(os.path.join(svdir, "weights"))
            if evals['train_loss'] < top_eval_loss:
                for file in weight_files: os.remove(os.path.join(svdir, "weights", file))
                U.save_full_model(model, None, svdir, ext='epoch%d_%.4f'%(epoch, evals['train_loss']))
                top_eval_loss = evals['train_loss']
            U.save_full_model(model, opt, svdir, ext='last')
        if msg:
            U.message_board(Line, "%s/epochout.txt"%svdir)
            allepochlines.append(Line)
            #save_train_loss(loss_list)
            #loss_list = []

        sys.stdout.write("\r\033[K%s"%Line)
        if msg:
            U.message_board(Line+'\n', "%s/epochout.txt"%svdir)
            allepochlines.append(Line+"\n")

    if msg:
        # Append results to the .all files
        U.message_board("".join(allepochlines), "save/epochout.all")
    
    print()

if __name__ == '__main__':
    if len(sys.argv)==1:
        train(master_config['epochs'])
    else:
        print("Running evaluation")
        with open("yaml/eval.yaml") as f:
            eval_config = yaml.safe_load(f)
        model.load_state_dict(th.load(eval_config['model_wts'], map_location=device))
        out = evaluation('test', save_df=True)
        out['dataframe'].to_parquet(eval_config['out_name'])
        print()
