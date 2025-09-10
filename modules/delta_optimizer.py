import torch
from torch.optim import Optimizer


"""
    该类包装了一个optimizer用于在step方法后返回optimizer优化前后的梯度变化值
"""
class DeltaOptimizer:
    def __init__(self, optimizer:Optimizer):
        self.optimizer = optimizer
        self.param_groups = self.optimizer.param_groups
        self._deltas = []

    def step(self):
        # step 前保存参数
        old_params = [p.clone().detach() for group in self.param_groups for p in group['params'] if p.requires_grad]

        self.optimizer.step()

        # step 后计算更新量
        new_params = [p.clone().detach() for group in self.param_groups for p in group['params'] if p.requires_grad]

        self._deltas = [new.detach() - old for old, new in zip(old_params, new_params)]

        return self._deltas

    def zero_grad(self, set_to_none: bool = False):
        self.optimizer.zero_grad(set_to_none=set_to_none)

    @property
    def deltas(self):
        """返回上一次 step 的 Δw 列表，对应 param_groups 中的参数顺序"""
        return self._deltas