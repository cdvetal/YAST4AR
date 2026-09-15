import time
from typing import List

import numpy as np
import torch


class Result:

    def __init__(self, batch: int, query_budget: int = np.inf):
        self.time_cost = 0
        self.query = np.zeros(batch, dtype=int)
        self.successed = np.zeros(batch, dtype=bool)
        self.distance = np.zeros(batch, dtype=float)
        self.budget = query_budget
        self.start_time = time.time()
        self.updated = False

    def update(self, success: torch.BoolTensor, query: int, distance: torch.tensor, hist=False):
        """
        update total query and success
        :param distance: distance to originals
        :param hist: flag to record hist
        :param query: number query at this update
        :param success: the success status of this candidate
        :return: None
        """
        _success = success.cpu().numpy() if query <= self.budget else np.zeros(success.shape, dtype=bool)
        self.updated = (~self.successed & _success).any()
        query = min(self.budget, query)
        # not success before but success this time should update query
        self.query[~self.successed] = query
        self.distance[~self.successed] = distance[~self.successed].detach().cpu().numpy()
        self.successed = self.successed | _success
        self.time_cost = time.time() - self.start_time

    def is_finish(self) -> bool:
        """
        decide whether all images attack success
        :return: bool
        """
        return self.successed.all().item() or (self.query >= self.budget).any()

    @property
    def get_successful_qd(self):
        return self.query[self.successed], self.distance[self.successed]

    def __repr__(self) -> str:
        value = f"@distance = {self.distance.mean().item():.4f} " \
                f"@query = {self.query.mean().item():.4f} " \
                f"@success = {self.successed.mean().item():.2%} " \
                f"@time cost = {self.time_cost:.4f}"
        return value

    @property
    def finished(self):
        return self.is_finish()
