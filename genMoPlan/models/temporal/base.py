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
        query_dim=0,
        query_length=0,
        **kwargs,
    ):
        super().__init__()

        self.prediction_length = prediction_length
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.query_dim = query_dim
        self.query_length = query_length

    @abstractmethod
    def forward(self, x, query, time):
        raise NotImplementedError("Subclasses must implement this method")
        