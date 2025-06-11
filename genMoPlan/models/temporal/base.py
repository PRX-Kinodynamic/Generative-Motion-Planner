from abc import abstractmethod
import torch
import torch.nn as nn
import math

from genMoPlan.models.helpers import SinusoidalPosEmb

class TemporalModel(nn.Module):
    def __init__(
        self, 
        prediction_length, 
        input_dim, 
        output_dim, 
        global_query_dim=0,
        global_query_length=0,
        local_query_dim=0,
        **kwargs,
    ):
        super().__init__()

        self.prediction_length = prediction_length
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.global_query_dim = global_query_dim
        self.global_query_length = global_query_length
        self.local_query_dim = local_query_dim

    @abstractmethod
    def forward(self, x, global_query=None, local_query=None, time=None):
        raise NotImplementedError("Subclasses must implement this method")
        