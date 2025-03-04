import torch
import torch.nn as nn

from .helpers import (
    SinusoidalPosEmb,
    Residual,
    PreNorm,
    LinearAttention,
)

class FeedforwardNN(nn.Module):
    def __init__(self,
                horizon,
                transition_dim,
                cond_dim,
                dim=32,
                dim_mults=(1, 2, 4, 8),
                attention=False, num_layers=6, hidden_dim: int = 128):
        super().__init__()

        self.transition_dim = transition_dim
        self.time_dim = horizon

        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(self.time_dim),
            nn.Linear(self.time_dim, self.time_dim * 4),
            nn.Mish(),
            nn.Linear(self.time_dim * 4, horizon),
        )

        layers = []

        layers.append(nn.Linear(transition_dim + 1, hidden_dim))
        layers.append(nn.Mish())

        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.Mish())

        if attention:
            layers.append(Residual(PreNorm(hidden_dim, LinearAttention(hidden_dim))))

        layers.append(nn.Linear(hidden_dim, transition_dim))

        self.model = nn.Sequential(*layers)

    def forward(self, x, time):
        """
        x : [ batch x horizon(length of trajectory) x transition_dim]
        time : [ batch_size ]
        """


        t = self.time_mlp(time)
        t=t.unsqueeze(2)

        # # breakpoint()
        t = t.expand((x.shape[0],x.shape[1],1))
        # breakpoint()
 
        x = torch.cat((x, t), dim=2)
        # breakpoint()
        return self.model(x)
        

        # return self.hidden_and_out_layers(x+t)

        # t = t.unsqueeze(1)
        # t = t.unsqueeze(2)
        # # t = t.unsqueeze(1).unsqueeze(2)
        # # t = t.expand(-1, x.shape[1], -1)
        # t = t.expand(x.shape[0], x.shape[1], -1)
        # merged = torch.cat((x, t), dim=2)
        # return self.model(merged)
