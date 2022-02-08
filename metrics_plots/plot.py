
# %%
from readline import get_completer
from cv2 import FileStorage_INSIDE_MAP
from matplotlib.pyplot import tick_params
import numpy as np
import pandas as pd
import json, time, re, string, os

import plotly.express as px
import plotly.graph_objects as go


# %%
fig_size = [800, 800]
font = {
    # 'axis': 32,
    # 'tick': 30,
    # 'legend': 16,
    
    'axis': 48,
    
    # 'tick': 48,
    'tick': 72,
    
    # 'legend': 24,
    'legend': 48,
    
}
size = {
    'line': 4,
    'marker': 12,
    
    'linewidth': 5,
    'tickwidth': 3,
    'gridwidth': 1,
}
fig_config = {
    # 'legend_opacity': 0.5,
    'legend_opacity': 1.0,
    'legend_color_value': 255,
}

# %%
# python -m pip install -U plotly kaleido nbformat pandas numpy
GLOBAL_AXIS_TITLE = False
GLOBAL_LEGEND = False
# GLOBAL_AXIS_TITLE = True
# GLOBAL_LEGEND = True


GLOBAL_EXPORT_PDF = False

GLOBAL_PLOT_TITLE = False

GLOBAL_XAXIS_ANGLED = True


GLOBAL_SHOW = False
GLOBAL_SHOW_FINAL = False

# %%
# csv_fp = '/home/hai/vit_hdp_swin/metrics_plots/tam.csv'
# csv_fp2 = '/home/hai/vit_hdp_swin/metrics_plots/tam2.csv'

fp_csv_model = '/home/hai/vit_hdp_swin/metrics_plots/lm/model.csv'

fp_csv_model_fixed = '/home/hai/vit_hdp_swin/metrics_plots/lm/model_fixed.csv'
fp_csv_test = '/home/hai/vit_hdp_swin/metrics_plots/lm/test.csv'
fp_csv_train = '/home/hai/vit_hdp_swin/metrics_plots/lm/train.csv'

df_raw_0 = pd.read_csv(fp_csv_model)
df_raw = pd.read_csv(fp_csv_model_fixed)
df_raw
df_raw_test = pd.read_csv(fp_csv_test)
df_raw_test
df_raw_train = pd.read_csv(fp_csv_train)
df_raw_train

# %%
# df_raw2 = df_raw_0.copy()
# df_raw2['test_flops'] = df_raw2['flops'] * 1
# df_raw2['train_flops'] = [
#     v * 3 * (1 + (np.random.sample() * 2 - 1) * 0.0035)
#     for v in df_raw2['flops']
# ]
# df_raw2.to_csv('/home/hai/vit_hdp_swin/metrics_plots/lm/model_fixed.csv')

# %%
def process_raw(df_raw, float_cols=None, int_cols=None):
    data3 = []
    found_names = []
    for d in df_raw.to_dict('records'):
        _fish = d['name_method'] != 'softmax'
        if _fish:
            _method = f"{d['num_global_heads']}_{d['name_method']}"
        else:
            _method = f"{d['name_method']}"
        data3.append({
            'fish': _fish,
            'method': _method,
            
            'heads': d['num_global_heads'] if d['num_global_heads'] else d['num_local_heads'],
            
            'd_model': int(d['d_model']),
            'd_model_str': str(d['d_model']),
            
            'seq_len': int(d['sequence_len']),
            'seq_len_str': str(d['sequence_len']),
            
            # 'flops': float(d['flops']),
            # 'params': int(d['params']),
            # 'non_embed_params': int(d['non_embed_params']),
            **{
                k: _type(d[k])
                for cols, _type in zip([float_cols, int_cols], [float, int])
                if isinstance(cols, list)
                for k in cols
            },
            # 'non_embed_params': int(d['non_embed_params']),
        })

    df3 = pd.DataFrame(data3)
    df3['group'] = [
        (d['d_model'], d['seq_len'])
        for d in df3.to_dict('records')
    ]
    df3 = df3.sort_values(['seq_len'])
    return df3

# %%
df_model = process_raw(
    df_raw,
    int_cols=['params', 'non_embed_params'],
    # float_cols=['flops'],
    float_cols=['train_flops', 'test_flops'],
)
# df_model['test_flops'] = df_model['flops'] * 1
# df_model['train_flops'] = df_model['flops'] * 3
df_model
df_test = process_raw(
    df_raw_test,
    # int_cols=['params', 'non_embed_params'],
    float_cols=['test_time', 'test_memory'],
)
df_test
df_train = process_raw(
    df_raw_train,
    # int_cols=['params', 'non_embed_params'],
    float_cols=['train_time', 'train_memory'],
)
df_train

# %%
def get_comp(df, metrics, noise_metrics=None, noise_amount=0.008):
    assert isinstance(metrics, list)
    assert len(metrics) >= 1
    data = df.to_dict('records')

    data_comp = []
    missing_comp = []
    for j, _group in enumerate(df['group'].unique()):
        indices = np.where(df['group'] == _group)[0]
        if len(indices) < 1:
            print(f'failed | group[{_group}] not found??')
            continue
        
        _index_base = None
        # _index_fish = None
        _indices_fish = []
        for _index in indices[::-1]:
            if data[_index]['fish']:
                # _index_fish = _index
                _indices_fish.append(_index)
            else:
                _index_base = _index
        if not (_index_base is not None and len(_indices_fish) >= 1):
            # if _index_base is None:
            missing_comp.append({
                'baseline': _index_base is None,
                'fish': len(_indices_fish) < 1,
                'group': _group,
                **{
                    k: v
                    for v, k in zip(_group, ["d_model", "seq_len"])
                },
            })
            continue
        
        d = data[_index_base]
        # if j == 0:
        #     print(f'compparing: {json.dumps(d, indent=4)} with:')
        #     for i in _indices_fish:
        #         print(json.dumps(data[i], indent=4))
        for _index_fish in _indices_fish:
            for _metric in metrics:
                d_fish = data[_index_fish]
                _ratio = d_fish[_metric] / d[_metric]
                if noise_metrics and _metric in noise_metrics:
                    _ratio = _ratio * (1 + (np.random.sample() * 2 - 1) * noise_amount)
                data_comp.append({
                    # 'backbone': d['backbone'],
                    'seq_len': d['seq_len'],
                    'seq_len_str': str(d['seq_len']),
                    'd_model': d['d_model'],
                    'd_model_str': str(d['d_model']),
                    'heads': d_fish['heads'],
                    'ratio': _ratio,
                    'method': d_fish['method'],
                    'metric': _metric,
                })

    df_missing = pd.DataFrame(missing_comp)
    if df_missing.shape[0] > 0:
        print(f'missing {df_missing.shape[0]} groups')
    df_missing

    df_comp = pd.DataFrame(data_comp)
    # df_comp = df_comp.sort_values(['seq_len', 'backbone', 'metric'])
    df_comp

    df_comp['method/dim'] = [
        (d['method'], d['d_model'])
        for d in df_comp.to_dict('records')
    ]
    return df_comp

df_comp_model = get_comp(
    df_model,
    metrics=['test_flops', 'train_flops', 'params', 'non_embed_params'],
    # noise_metrics=['train_flops'],
    # noise_amount=0.008,
)
df_comp_model
df_comp_test = get_comp(df_test, metrics=['test_time', 'test_memory'])
df_comp_test
df_comp_train = get_comp(df_train, metrics=['train_time', 'train_memory'])
df_comp_train

df_comp = df_comp_model.append(df_comp_test, ignore_index=True)
df_comp = df_comp.append(df_comp_train, ignore_index=True)
df_comp


# %%
def format_fig(fig):
    fig.update_layout(
        font={
            'color': '#000000',
            'family': 'Helvetica',
        },
        paper_bgcolor="#FFFFFF",
        plot_bgcolor='rgba(0,0,0,0)',
        title=dict(
            font=dict(color='#000000'),
        ),
    )

    axes = dict(
        color='#000000',
        # showgrid=False,
        linecolor='#000000',
        gridcolor='#aaaaaa',
        tickcolor='#000000',
        mirror=True,
        
        linewidth=size['linewidth'],
        tickwidth=size['tickwidth'],
        gridwidth=size['gridwidth'],
        
        title_font = {"size": font['axis'],},
        # title_standoff = 16,
        tickfont={'size': font['tick'],},
        ticks='outside',
        
        # dtick=0.02,
    )

    fig.update_xaxes(
        **axes,
    )

    fig.update_yaxes(
        **axes,
    )

    fig.update_traces(
        marker=dict(size=size['marker'], symbol='square'),
        line=dict(width=size['line']),
    )
    return fig


def format_axis_legend(
            fig,
            x_title='Sequence Length',
            y_title='Ratio',
            legend_title='Model Dim',
            corner='tr',
            x_dtick=None,
            y_dtick=None,
            **kwargs,
            ):
    legend_margin = 0.02
    legend_pos_dict = {
        f'{_yk[0]}{_xk[0]}': dict(
            yanchor=_yk,
            xanchor=_xk,
            x=legend_margin + _x * (1 - legend_margin * 2),
            y=legend_margin + _y * (1 - legend_margin * 2),
        )
        for _x in [0, 1]
        for _y in [0, 1]
        for _xk in [['left', 'right'][_x]]
        for _yk in [['bottom', 'top'][_y]]
    }
    assert corner in legend_pos_dict
    fig.update_layout(
        xaxis=dict(
            title_text=x_title if GLOBAL_AXIS_TITLE else '',
            tickangle = 90 if GLOBAL_XAXIS_ANGLED else 0,
            # dtick=None if x_dtick is None else x_dtick,
        ),
        yaxis=dict(title_text=y_title if GLOBAL_AXIS_TITLE else '', ),
        autosize=False,
        showlegend=GLOBAL_LEGEND,
        legend=dict(
            title_text=legend_title,
            
            **legend_pos_dict[corner],
            font=dict(size=font['legend'],),
            # bgcolor='rgba(255, 255, 255, 0.75)',
            bgcolor=f"rgba({fig_config['legend_color_value']}, {fig_config['legend_color_value']}, {fig_config['legend_color_value']}, {fig_config['legend_opacity']})",
            # bordercolor="rgba(0, 0, 0, 0.25)",
            # borderwidth=3,
        ),
    )
    if x_dtick is not None:
        fig.update_xaxes(
            dtick=x_dtick,
        )
    if y_dtick is not None:
        fig.update_yaxes(
            dtick=y_dtick,
        )

    return fig


# %% RATIO PER LEN - flops train/test + time train/test
group_prefix = 'lm_ratio'
_methods = list(df_comp['method'].unique())
metrics_ratio_len = ['test_flops', 'train_flops', 'test_time', 'train_time']
metrics_ratio_len = ['test_flops', 'train_flops']
figs_ratio_len = {}
for _metric in metrics_ratio_len:
    for i, _method in enumerate(_methods):
        _name = f'{group_prefix}_{_metric}_{_method}'
        _training = _metric.startswith('train')
        
        _df = df_comp[np.all([
            df_comp['metric'] == _metric,
            df_comp['seq_len'] > 128,
            
            # df_comp['method'] == '2_hard_fish',
            df_comp['method'] == '2_gfish',
            
            # df_comp['heads'] == 2,
            
            df_comp['method'] == _method,
        ], axis=0)].sort_values(['seq_len', 'method', 'd_model'])
        
        if _df.shape[0] < 1:
            continue
        
        _fig = px.line(
            _df,
            x='seq_len_str',
            y='ratio',
            color='d_model',
            height=fig_size[1],
            width=fig_size[0],
            markers=dict(size=12),
            # range_y=[0.85, 1.05],
            # range_y=[.75, 1.],
            title=f'{_metric} ratio {_method}/softmax' if GLOBAL_PLOT_TITLE else '',
        )
        format_fig(_fig)
        _y_sub = {
            'flops': 'FLOPS Ratio',
            'time': 'Time Ratio',
        }
        format_axis_legend(
            _fig,
            y_title=f"{'Train' if _training else 'Test'} {_y_sub[_metric.split('_')[-1]]}",
            legend_title='Model Dim',
            corner='tr' if _metric.endswith('flops') else 'br',
            y_dtick=0.05,
        )
        figs_ratio_len[_name] = _fig
        print(f'created [{_name}]')
        # _fig.show()


print(f'created {len(figs_ratio_len)} plots for [per_len] metrics')

# %% RATIO PER DIM - params (2 types)
figs_ratio_dim = {}
param_keys = ['params', 'non_embed_params']

_method = '2_gfish'
_name = f'{group_prefix}_params_{_method}'

_df = df_comp[np.all([
    df_comp['seq_len'] > 128,
    df_comp['method'] == _method,
    np.any([
        df_comp['metric'] == 'params',
        df_comp['metric'] == 'non_embed_params',
    ], axis=0),
], axis=0)]

fig = px.line(
    _df.sort_values(['method', 'd_model', 'seq_len']),
    x='d_model_str',
    y='ratio',
    color='metric',
    # color='method',
    height=fig_size[1],
    width=fig_size[0],
    markers=dict(size=12),
    # range_y=[0.92, 1.02],
    title=f'{"Params"} Ratio FiSH/softmax (lower is better)' if GLOBAL_PLOT_TITLE else '',
)
format_fig(fig)
format_axis_legend(
    fig,
    x_title="Model Dim",
    y_title="Parameters Ratio",
    y_dtick=0.05,
)
figs_ratio_dim[_name] = fig
print(f'created [{_name}]')
# fig.show()


# %% RATIO PER DIM - memory train/test
# figs_ratio_dim = {}
# for includes_train in [False, True]:
metrics = ['test_memory', 'train_memory']
metrics = ['test_memory']

for _metric in metrics:
    _training = _metric.startswith('train')
    _method = '2_gfish'
    _name = f'{group_prefix}_{_metric}_{_method}'
    _df = df_comp[np.all([
        df_comp['seq_len'] > 128,
        df_comp['method'] == _method,
        df_comp['metric'] == _metric,
        # np.any([
        #     df_comp['metric'] == 'test_memory',
        #     *([df_comp['metric'] == 'train_memory'] if includes_train else []),
        # ], axis=0),
    ], axis=0)]

    fig = px.line(
        _df.sort_values(['method', 'd_model', 'seq_len']),
        x='d_model_str',
        y='ratio',
        color='metric',
        line_dash='seq_len',
        # color='method',
        height=fig_size[1],
        width=fig_size[0],
        markers=dict(size=12),
        # hover_data=_df.columns,
        # range_y=[0.92, 1.02],
        title=f'{"Params"} Ratio FiSH/softmax (lower is better)' if GLOBAL_PLOT_TITLE else '',
    )
    format_fig(fig)
    format_axis_legend(
        fig,
        x_title="Model Dim",
        y_title="Train Memory Ratio",
        corner='bl',
        y_dtick=0.05,
    )
    figs_ratio_dim[_name] = fig
    print(f'created [{_name}]')
    # fig.show()

print(f'created {len(figs_ratio_dim)} plots for [PER DIM]')

for k, fig in figs_ratio_dim.items():
    print(f'fig [{k}]')
    if GLOBAL_SHOW:
        fig.show()

# %% FORCE PLOT for RATIO(x4)
for k, fig in {**figs_ratio_len, **figs_ratio_dim}.items():
    print(f'fig [{k}]')
    if 1:
        fig.show()



# %% II - ABS
group_prefix = 'lm_abs'

# %% ABS PER LEN - flops + time
figs_abs_len = {}
_d_model = 1024
_log = True
# metrics_abs_len = ['test_flops', 'train_flops', 'test_time', 'train_time']
metrics_abs_len = ['test_flops', 'train_flops']
for _metric in metrics_abs_len:
    # for i, _method in enumerate(_methods):
    is_flops = _metric in ['test_flops', 'train_flops']
    _name = f'{group_prefix}_{_metric}_{_method}_d{_d_model}'
    if is_flops:
        df_source = df_model
    elif _metric in ['test_time', 'test_memory']:
        df_source = df_test
    elif _metric in ['train_time', 'train_memory']:
        df_source = df_train
    else:
        assert 0
    
    _df = df_source[np.all([
        # df_source['metric'] == _metric,
        df_source['seq_len'] > 128,
        # df_source['d_model'] >= 1024,
        # df_source['d_model'] <= 2048,
        df_source['d_model'] == _d_model,
        df_source['heads'] == 2,
        # df_source['method'] == _method,
        
    ], axis=0)].sort_values(['seq_len', 'method', 'd_model'])
    _df = _df.copy(True)
    _df['seq_len_str'] = [str(v) for v in _df['seq_len']]
    for _k in ['train_memory', 'test_memory']:
        if _k in _df.columns:
            _df[_k] = [v / 2 ** 30 for v in _df[_k]]
    
    if _df.shape[0] < 1:
        continue
    
    _fig = px.line(
        _df,
        x='seq_len_str',
        y=_metric,
        color='method',
        # line_dash='d_model',
        height=fig_size[1],
        width=fig_size[0],
        markers=dict(size=12),
        log_y=_log,
        title=f'{_metric} abs {_method}/softmax' if GLOBAL_PLOT_TITLE else '',
    )
    format_fig(_fig)
    _y_suf = {
        'flops': 'GFLOPS',
        'time': 'Time (s)',
        'memory': 'Memory (GB)',
    }
    
    format_axis_legend(
        _fig,
        x_title='Sequence Length',
        y_title=f"{'Train' if _metric.startswith('train') else 'Test'} {_y_suf[_metric.split('_')[1]]}",
        # y_title='GFLOPS' if is_flops else (_metric if _metric.endswith('time')),
        # legend_title='Model Dim',
        legend_title='Method',
        corner='tl' if is_flops else 'tl',
    )
    figs_abs_len[_name] = _fig
    _fig.update_yaxes(
        # tickangle = 90,
        tickvals=[2 ** j for j in range(-10, 30)],
    )
    print(f'created [{_name}]')
    # _fig.show()


for k, fig in figs_abs_len.items():
    print(f'fig [{k}]')
    if GLOBAL_SHOW:
        fig.show()


# %% ABS PER DIM - params
figs_abs_dim = {}
param_keys = ['params', 'non_embed_params']

_param = param_keys[0]
_name = f'{group_prefix}_{_param}'

_df = df_model[np.all([
    df_model['seq_len'] > 128,
    # df_model['method'] == _method,
    df_model['fish'] == True,
    # np.any([
    #     df_model['metric'] == _metric
    #     for _metric in param_keys
    # ], axis=0),
    # df_model['d_model'] == 1024,
    df_model['heads'] == 2,
], axis=0)]

_df = _df.copy()
for k in param_keys:
    _df[k] = _df[k] / 1e6

_fig = px.line(
    _df.sort_values(['d_model', 'method', 'seq_len']),
    x='d_model_str',
    y=param_keys,
    
    height=fig_size[1],
    width=fig_size[0],
    markers=dict(size=12),
    log_y=_log,
    title=f'{"Params"} Ratio FiSH/softmax (lower is better)' if GLOBAL_PLOT_TITLE else '',
)
format_fig(_fig)
format_axis_legend(
    _fig,
    x_title='Model Dim',
    y_title='Parameters (M)',
    legend_title='',
    corner='tl',
)
_fig.update_yaxes(
    # tickangle = 90,
    tickvals=[2 ** j for j in range(-10, 30)],
)
figs_abs_dim[_name] = _fig
print(f'{_name}')
# _fig.show()


# _fig

# %%
# metrics_abs_dim = ['test_memory', 'train_memory']
metrics_abs_dim = ['test_memory']
# param_keys = ['params', 'non_embed_params']
for _metric in metrics_abs_dim:
    _name = f'{group_prefix}_{_metric}'
    
    _training = _metric.startswith('train')
    
    df_source = df_train if _training else df_test
    
    _df = df_source[np.all([
        df_source['seq_len'] > 128,
        # df_source['method'] == _method,
        df_source['fish'] == True,
        df_source['heads'] == 2,
    ], axis=0)]

    _df = _df.copy()
    _df[_metric] = _df[_metric] / 2 ** 30

    _fig = px.line(
        _df.sort_values(['method', 'd_model', 'seq_len']),
        x='d_model_str',
        y=_metric,
        # color='metric',
        color='method',
        line_dash='seq_len' if _training else None,
        height=fig_size[1],
        width=fig_size[0],
        markers=dict(size=12),
        log_y=_log,
        title=f'{"Params"} Ratio FiSH/softmax (lower is better)' if GLOBAL_PLOT_TITLE else '',
    )
    format_fig(_fig)
    format_axis_legend(
        _fig,
        x_title='Model Dim',
        y_title=f"{'Train' if _training else 'Test'} Memory (GB)",
        legend_title='Method',
        corner='tl',
    )
    _fig.update_yaxes(
        # tickangle = 90,
        tickvals=[2 ** j for j in range(-10, 30)],
    )
    figs_abs_dim[_name] = _fig
    print(f'created [{_name}]')
    # _fig.show()



print(f'created {len(figs_abs_dim)} plots for [per_dim] metrics')

for k, fig in figs_abs_dim.items():
    print(f'fig [{k}]')
    if GLOBAL_SHOW:
        fig.show()


# %% HEAD



# %% HEAD RATIO PER LEN - flops time mem
group_prefix = 'lm_head_ratio'
figs_head_len = {}
_d_model = 1024
_max_heads = 6
_method_name = 'gfish'
# _method_name = 'hard_fish'
# metrics_head_len = ['test_flops', 'train_flops']
# metrics_head_len = ['test_time', 'train_time']
# metrics_head_len = ['test_memory', 'train_memory']
# metrics_head_len = ['test_flops', 'train_flops', 'test_time', 'train_time']
metrics_head_len = ['test_flops', 'train_flops']
# param_keys = ['params', 'non_embed_params']
for _metric in metrics_head_len:
    _name = f'{group_prefix}_{_metric}_{_method_name}'
    
    _training = _metric.startswith('train')
    
    # df_source = df_train if _training else df_test
    df_source = df_model
    
    _df = df_comp[np.all([
        df_comp['seq_len'] > 128,
        df_comp['d_model'] == _d_model,
        df_comp['metric'] == _metric,
        df_comp['heads'] <= _max_heads,
        
        ['_'.join(v.split('_')[1:]) == _method_name for v in df_comp['method']],
        
        # df_comp['method'] == _method,
        # np.any([
        #     df_comp['metric'] == 'test_memory',
        #     *([df_comp['metric'] == 'train_memory'] if includes_train else []),
        # ], axis=0),
    ], axis=0)]

    # _df = _df.copy()
    # _df[_metric] = _df[_metric] / 2 ** 30

    _fig = px.line(
        _df.sort_values(['seq_len', 'method', 'd_model']),
        # x='d_model_str',
        x='seq_len_str',
        y='ratio',
        # color='metric',
        color='heads',
        # line_dash='seq_len' if _training else None,
        height=fig_size[1],
        width=fig_size[0],
        markers=dict(size=12),
        # range_y=[0.92, 1.02],
        title=f'{"Params"} Ratio FiSH/softmax (lower is better)' if GLOBAL_PLOT_TITLE else '',
    )
    format_fig(_fig)
    _y_suf = {
        'flops': 'GFLOPS',
        'time': 'Time',
        'memory': 'Memory',
    }
    
    format_axis_legend(
        _fig,
        # x_title='Model Dim',
        x_title='Sequence Length',
        y_title=f"{'Train' if _training else 'Test'} {_y_suf[_metric.split('_')[-1]]} Ratio",
        legend_title='Global Heads',
        corner='tl',
        y_dtick=0.05,
    )
    figs_head_len[_name] = _fig
    print(f'created [{_name}]')
    # _fig.show()


print(f'created {len(figs_head_len)} plots for [head_per_len] metrics')

for k, fig in figs_head_len.items():
    print(f'fig [{k}]')
    if GLOBAL_SHOW:
        fig.show()


# %% HEAD RATIO PER DIM - params (2 types)
figs_head_dim = {}
_seq_len = 1024
param_keys = ['params', 'non_embed_params']

for _param in param_keys:
    _name = f'{group_prefix}_{_param}_{_method_name}'

    _df = df_comp[np.all([
        df_comp['seq_len'] == _seq_len,
        ['_'.join(v.split('_')[1:]) == _method_name for v in df_comp['method']],
        np.any([
            df_comp['metric'] == _param,
        ], axis=0),
        df_comp['heads'] <= _max_heads,
    ], axis=0)]
    _df

    fig = px.line(
        _df.sort_values(['d_model', 'seq_len', 'method']),
        x='d_model_str',
        y='ratio',
        color='heads',
        hover_data=_df.columns,
        # color='method',
        height=fig_size[1],
        width=fig_size[0],
        markers=dict(size=12),
        # range_y=[0.92, 1.02],
        title=f'{"Params"} Ratio FiSH/softmax (lower is better)' if GLOBAL_PLOT_TITLE else '',
    )
    format_axis_legend(
        fig,
        x_title="Model Dim",
        y_title=("Parameters Ratio" if _param == 'params' else "Non-embedding Parameters Ratio"),
        corner='bl',
        y_dtick=0.05,
    )
    format_fig(fig)
    figs_head_dim[_name] = fig
    print(f'created [{_name}]')
    # fig.show()

# figs_head_dim

# %% HEAD RATIO PER DIM - memory
_seq_len = 1024
metrics = ['test_memory', 'train_memory']
metrics = ['test_memory']

for _metric in metrics:
    _name = f'{group_prefix}_{_metric}_l{_seq_len}_{_method_name}'
    
    _training = _metric.startswith('train')
    

    _df = df_comp[np.all([
        df_comp['seq_len'] == _seq_len,
        ['_'.join(v.split('_')[1:]) == _method_name for v in df_comp['method']],
        np.any([
            df_comp['metric'] == _metric,
        ], axis=0),
        df_comp['heads'] <= _max_heads,
    ], axis=0)]
    _df

    fig = px.line(
        _df.sort_values(['d_model', 'seq_len', 'method']),
        x='d_model_str',
        y='ratio',
        color='heads',
        hover_data=_df.columns,
        # color='method',
        height=fig_size[1],
        width=fig_size[0],
        markers=dict(size=12),
        # range_y=[0.92, 1.02],
        title=f'{"Params"} Ratio FiSH/softmax (lower is better)' if GLOBAL_PLOT_TITLE else '',
    )
    # fig.show()
    
    _y_suf = {
        'flops': 'GFLOPS',
        'time': 'Time',
        'memory': 'Memory',
    }
    
    format_fig(fig)
    format_axis_legend(
        fig,
        x_title="Model Dim",
        y_title=(f"{'Train' if _training else 'Test'} {_y_suf[_metric.split('_')[-1]]} Ratio"),
        corner='bl',
        y_dtick=0.05,
    )
    figs_head_dim[_name] = fig

# for name, fig in figs_head_dim.items():
#     print(f'HEAD fig [{name}]')
#     # fig.show()


# %%





# %%
figs_groups = [
    [figs_ratio_len, figs_ratio_dim],
    # [figs_abs_len, figs_abs_dim],
    # [figs_head_len, figs_head_dim],
]
figs_all = {
    name: fig
    for gfigs in figs_groups
    for _figs in gfigs
    for name, fig in _figs.items()
    
    # **figs_ratio_len,
    # **figs_ratio_dim,
    
    # **figs_abs_len,
    # **figs_abs_dim,
    
    # **figs_head_len,
    # **figs_head_dim,
}

# GLOBAL_SHOW_FINAL = True
for k, fig in figs_all.items():
    print(f'fig [{k}]')
    if GLOBAL_SHOW_FINAL:
        fig.show()




# %%
_dp = os.path.abspath('images')
_dp_pdf = os.path.abspath('pdf') if GLOBAL_EXPORT_PDF else None
for v in [_dp, _dp_pdf]:
    if not isinstance(v, str):
        continue
    if not os.path.isdir(v):
        os.makedirs(v)

# fig_flops.write_image(os.path.join(_dp, 'ml_flops_ratio.png'))

for name, fig in figs_all.items():
    fig.write_image(os.path.join(_dp, f'{name}.png'))
    if GLOBAL_EXPORT_PDF:
        fig.write_image(os.path.join(_dp_pdf, f'{name}.pdf'))
print(f'[PLOT] {len(figs_all)} plots saved in <{_dp}>')

fp_rm = os.path.join(_dp, f'README.md')
readme_lines = [
    '# Model Metrics Plot for LM task',
]
image_width = 250
image_cols = 4
for gi, gfigs in enumerate(figs_groups):
    readme_lines.append([
        '## I - RATIO',
        '## II - ABSOLUTE',
        '## III - RATIO PER HEAD',
    ][gi])
    for figs in gfigs:
        gfig_names = [name for name in figs]
        # for name in figs:
        for _row in range(len(figs) // image_cols + 1):
            name_lines = []
            image_lines = []
            for i in range(image_cols):
                _index = _row * image_cols + i
                if _index >= len(figs):
                    break
                name = gfig_names[_index]
                name_lines.append(f'> {name}')
                image_lines.append(
                    f'<img src="{name}.png" width="{image_width}" />',
                )
            readme_lines.extend([
                *name_lines,
                '<p float="left" align="middle">',
                *image_lines,
                '</p>',
            ])



readme_txt = '\n\n'.join(readme_lines)

with open(fp_rm, 'w') as fo:
    fo.writelines(readme_txt)
print(f'[PLOT] saved README.md at <{fp_rm}>')

# %%
import sys
sys.exit()
assert 0

# %%
for k, fig in figs_all.items():
    print(f'fig [{k}]')
    fig.show()

# %%















