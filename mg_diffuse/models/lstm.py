import torch
import torch.nn as nn

from .helpers import SinusoidalPosEmb

class LSTMModel(nn.Module):
    # def __init__(self, input_dim=1, hidden_dim=64, num_layers=2, output_dim=1):
    def __init__(self,
                horizon,
                transition_dim,
                cond_dim,
                dim=32,
                dim_mults=(1, 2, 4, 8),
                attention=False, time_dim: int = 1, num_layers=6, hidden_dim: int = 128):
        
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.transition_dim = transition_dim
        self.time_dim = horizon

        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(self.time_dim),
            nn.Linear(self.time_dim, self.time_dim * 4),
            nn.Mish(),
            nn.Linear(self.time_dim * 4, horizon),
        )

        self.lstm = nn.LSTM(transition_dim + time_dim, hidden_dim, num_layers, batch_first=True)


        self.fc = nn.Linear(hidden_dim, transition_dim)

    def forward(self, x, time):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)

        t = self.time_mlp(time)

        t=t.unsqueeze(2)
        t = t.expand((x.shape[0],x.shape[1],1))

        x = torch.cat((x, t), dim=2)

        out, _ = self.lstm(x, (h0, c0))  
        return self.fc(out)