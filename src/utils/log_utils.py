from collections import deque, defaultdict
import torch

class SmoothedValue:
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size = 20, fmt = None):
        if fmt is None:
            fmt = "{value:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen = window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n = 1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]
    
    @property
    def sum(self):
        return self.total

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)

class MetricLogger:
    def __init__(self, delimiter="\t", window_size: int = 20, fmt = None):
        self.meters = defaultdict(lambda: SmoothedValue(window_size, fmt))
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        meter_str = []
        for name, meter in self.meters.items():
            meter_str.append(f"    {name}: {meter}")
        return self.delimiter.join(meter_str)
    
    def add_meter(self, name, meter):
        self.meters[name] = meter

class OutputWriter():

    def __init__(self, log_file: str):
        self.log_file = log_file

    def _log(self, info: str):
        with open(self.log_file, 'a') as f:
            f.write(info)
    
    def _write(self, data, level):
        if isinstance(data, dict):
            for key, value in data.items():
                if level == 0:  
                    self._log(f'{key}: ')  
                else:
                    self._log('\n' + '    ' * level + f'{key}: ')
                self._write(value, level + 1)  
        elif isinstance(data, list):
            for item in data:
                self._log('\n' + '    ' * level + '- ' + str(item))  
        else:
            self._log(str(data))  


    def write_dict(self, data, level=0):
        self._write(data, level)
        self._log('\n')


