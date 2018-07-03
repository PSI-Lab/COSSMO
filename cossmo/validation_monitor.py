import numpy as np


class ValidationMonitor(object):
    def __init__(self, maximize_metric=True, delay_steps=3, prev_hwm=None):
        self.maximize_metric = maximize_metric
        self.delay_steps = delay_steps
        self.past_metrics = []
        self.hwm = (-np.inf, 0) if maximize_metric else (np.inf, 0)
        self._is_hwm = False
        if prev_hwm and prev_hwm[1]>0:
            self.hwm = prev_hwm

    def is_hwm(self):
        return self._is_hwm

    def __call__(self, metric_val, global_step):
        self.past_metrics.append((metric_val, global_step))

        if len(self.past_metrics) >= self.delay_steps:
            self.past_metrics = self.past_metrics[-self.delay_steps:]

        if (self.maximize_metric and metric_val > self.hwm[0]) or \
           (not self.maximize_metric and metric_val < self.hwm[0]) or\
                (self.hwm[1]==0):

            self.hwm = (metric_val, global_step)
            self._is_hwm = True
            print("setting high point at",global_step,metric_val)
        else:
            self._is_hwm = False

    def should_stop(self):
        # Don't stop training until at least self.delay_stops data points are available
        if len(self.past_metrics) < self.delay_steps:
            return False

        first_diff = [b[0] - a[0] for a, b in zip(self.past_metrics[:-1], self.past_metrics[1:])]

        if self.maximize_metric and all(d < 0 for d in first_diff):
            return True
        elif not self.maximize_metric and all(d > 0 for d in first_diff):
            return True
        else:
            return False
