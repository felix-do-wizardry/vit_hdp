# %%
import os, time
import argparse
import glob
import re
import json
import pandas as pd
from shutil import copyfile

# %%
args = {
    'output': 'output',
    'keep_count': 1,
}

# %%
parser = argparse.ArgumentParser('...')
parser.add_argument('--output', default='output', type=str)
parser.add_argument('--keep_count', default=1, type=int)
# parser.add_argument('--output', default='output', type=str)
# parser.add_argument('--output', default='output', type=str)
# parser.add_argument('--output', default='output', type=str)
args = parser.parse_args().__dict__

print(args)

# %%
assert args['keep_count'] == 1

# %%
files_config = glob.glob(f"{args['output']}/**/config.json", recursive=True)
files_log = glob.glob(f"{args['output']}/**/log_rank0.txt", recursive=True)
files_pth = glob.glob(f"{args['output']}/**/*.pth", recursive=True)

dirs = {}
for _type, _files in zip(['config', 'log', 'pth'], [files_config, files_log, files_pth]):
    for v in _files:
        _dir, _name = os.path.split(v)
        if _dir not in dirs:
            dirs[_dir] = {
                'config': None,
                'log': None,
                'pth': [],
            }
        if _type in ['pth']:
            dirs[_dir][_type].append(v)
        else:
            assert dirs[_dir][_type] is None
            dirs[_dir][_type] = v

list(dirs.keys())

# %%
dirs_list = list(dirs.values())


# %%
def process(_dir):
    # a = dirs_list[0]['log']
    _logs = None
    with open(_dir['log'], 'r') as fo:
        _logs = list(fo)

    len(_logs)

    groups = []
    for _txt in _logs:
        g = [v.group(1) for v in re.finditer('.+WandB.+(\{.+val.+\})', _txt)]
        if len(g) > 0:
            groups.append(g[0])

    len(groups), groups[:5]

    logs_dict = [
        json.loads(g.replace("'", '"'))
        for g in groups
    ]
    len(logs_dict), logs_dict[:5]
    
    pth_dict = {}
    for v in _dir['pth']:
        fn = os.path.split(v)[1]
        if fn.startswith('ckpt_epoch_') and fn.endswith('.pth'):
            _epoch = int(fn[len('ckpt_epoch_') : - len('.pth')])
            pth_dict[_epoch] = v
    
    df_logs = pd.DataFrame([
        {
            'epoch': i,
            **v,
            'pth': pth_dict.get(i, None),
        }
        for i, v in enumerate(logs_dict)
    ])
    df_logs
    if df_logs.shape[0] <= 0:
        return []
    # return df_logs
    
    epochs = df_logs.to_dict('records')
    epochs = [v for v in epochs if v['pth'] is not None]
    epochs = sorted(epochs, key=lambda v: v['val_acc'], reverse=True)
    return epochs
    # for i, d in enumerate():
    

# df = process(dirs_list[0])
# df
# r = process(dirs_list[0])
# r[:5]

# %%
for _dir in dirs_list:
    r = process(_dir)
    for v in r[1:]:
        _fp = v['pth']
        if os.path.isfile(_fp):
            os.remove(_fp)
    if len(r) > 0:
        _fp = r[0]['pth']
        _fp_best = os.path.join(os.path.split(r[0]['pth'])[0], 'ckpt_best.pth')
        copyfile(_fp, _fp_best)
        

# %%
'2021-12-29 16:30:33 swin_tiny_patch4_window7_224_hdp2qk_nonlinear] (main.py 329): INFO  * Acc@1 2.696 Acc@5 8.990'
"[2021-12-29 16:30:33 swin_tiny_patch4_window7_224_hdp2qk_nonlinear] (main.py 191): INFO [WandB]: {'val_loss': 6.15138460723877, 'val_acc': 2.696, 'val_acc5': 8.99}"


