# %%
# import numpy as np
# import pandas as pd
import time

# %%
class Timing:
    def __init__(self, allow_parallel=False, back_log=100):
        self.comps = {}
        self.running = True
        # self.allow_parallel = bool(allow_parallel)
        assert allow_parallel is False, '`allow_parallel` is currently not supported'
        self.allow_parallel = False
        self.time_start = self.time()
        self.agg = {
            'time': {
                '__total': 0.,
            },
            'ratio': {
                '__total': 1.,
            },
        }
        self.back_log = int(back_log)
    
    def reset(self):
        self.comps = {}
        # self.running = True
        self.time_start = self.time()
        self.agg = {
            'time': {
                '__total': 0.,
            },
            'ratio': {
                '__total': 1.,
            },
        }
    
    @classmethod
    def time(cls):
        return time.perf_counter()
        # return time.time()
    
    def update(self):
        time_current = self.time()
        _agg = {
            'time': {
                '__total': 0.,
                # '__other': 0.,
            },
            'ratio': {
                '__total': 1.,
                # '__other': 0.,
            },
        }
        _total_elapsed = 0.
        _first_start = time_current
        _last_finish = float(self.time_start)
        for _name, _comp in self.comps.items():
            # _comp['elapsed'] = sum([(v[1] if v[1] is not None else time_current) - v[0] for v in _comp['periods']])
            _agg['time'][_name] = float(_comp['elapsed'])
            _agg['time']['__total'] += float(_comp['elapsed'])
            
            if _first_start > _comp['start']:
                _first_start = float(_comp['start'])
            if not _comp['stopped']:
                _last_finish = float(time_current)
            elif _comp['finish'] is not None and _last_finish < _comp['finish']:
                _last_finish = float(_comp['finish'])
        
        for _name, _comp in self.comps.items():
            _agg['ratio'][_name] = _agg['time'][_name] / max(_agg['time']['__total'], 0.000000001)
        
        self.agg = _agg
        return _agg
    
    def tick(self):
        pass
    
    def pause(self, name=None, update=False):
        time_current = self.time()
        self.running = False
        
        if isinstance(name, str) and name in self.comps:
            _comps = {name: self.comps[name]}
        else:
            _comps = self.comps
        
        for _name, _comp in _comps.items():
            if _comp['stopped']:
                pass
            else:
                _comp['stopped'] = True
                _comp['finish'] = float(time_current)
                _comp['periods'][-1][1] = float(time_current)
                if self.back_log > 0:
                    _comp['periods'] = _comp['periods'][-self.back_log:]
                # _comp['elapsed'] += _comp['periods'][-1][1] - _comp['periods'][-1][0]
                _comp['elapsed'] += _comp['periods'][-1][1] - _comp['periods'][-1][0]
                _comp['elapsed_count'] += 1
        
        if update:
            self.update()
    
    def start_comp(self, name=None, pause_others=True, update=False):
        time_current = self.time()
        assert self.allow_parallel or pause_others
        assert isinstance(name, str)
        
        if pause_others:
            self.pause()
        
        if name in self.comps:
            _comp = self.comps[name]
            if not _comp['stopped']:
                self.pause(name)
            _comp['stopped'] = False
            _comp['periods'].append([
                time_current,
                None,
            ])
            _comp['period_count'] += 1
            # else:
            #     # _comp['stopped'] = True
            #     # _comp['periods'][-1][1] = time_current
            #     self.pause(name)
        else:
            if pause_others:
                self.pause()
            self.comps[name] = {
                'start': time_current,
                'finish': None,
                'elapsed': 0.,
                'elapsed_count': 0,
                'periods': [[float(time_current), None]],
                'period_count': 1,
                'stopped': False,
            }
            self.running = True
        
        if update:
            self.update()
    
    def __call__(self, name=None, pause_others=True, update=False,):
        return self.start_comp(name=name, pause_others=pause_others, update=update)
    
    def __repr__(self) -> str:
        return f'Timing({len(self.comps)} comps, {self.agg["time"]["__total"]} total secs)'
    
    @property
    def agg_str(self):
        return ' | '.join([
            f'{k}[{self.agg["time"][k]:.1f}s/{self.agg["ratio"][k]:.1%}]'
            for k in self.agg['time']
        ])
    
    @property
    def agg_table(self):
        row_strs = ['', 's', '%']
        header_len = max([len(v1) for v1 in row_strs])
        _table = [[v.rjust(header_len)] for v in row_strs]
        for k in self.agg['time']:
            if k not in self.agg['ratio']:
                continue
            _str_table_col = [str(k), f'{self.agg["time"][k]:.1f}s', f'{self.agg["ratio"][k] * 100:.2f}%']
            _col_width = max([len(v) for v in _str_table_col])
            _str_table_col = [f'{v:>{_col_width}}' for v in _str_table_col]
            for i, v in enumerate(_str_table_col):
                _table[i].append(v)
        
        return _table
    
    def agg_table_str(self, sep=' | ', indent=0):
        _agg_table = self.agg_table
        return '\n'.join([
            f'{" " * indent}{sep.join(_row)}'
            for _row in _agg_table
        ])

# %%
if __name__ == '__main__':
    T = Timing()
    
    dist = [2, 4, 1, 3]
    for i in range(200):
        for j, v in enumerate(dist):
            T(str(j))
            time.sleep(v / sum(dist) * 0.015)
    
    T.pause(update=True)
    # T.update()
    
    # print(json.dumps(T.agg, indent=4))
    print(T.agg_str)
    

# %%