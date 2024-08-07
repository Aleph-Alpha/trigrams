from typing import Optional, Union
import math

import torch

from ..config import TransformerArchitectureConfig


class MLP(torch.nn.Module):
    def __init__(
        self,
        config: TransformerArchitectureConfig,
        layer_index: int,
        device: Optional[Union[str, torch.device]],
    ):
        super().__init__()
        self.config = config
        self.layer_index = layer_index

        self.dense_in = torch.nn.Linear(
            in_features=self.config.hidden_size,
            out_features=int(self.config.hidden_size * self.config.mlp_factor),
            dtype=self.config.precision.dtype,
            device=device,
            bias=self.config.bias_terms,
        )

        if self.config.mlp_type == "swiglu":
            self.swiglu = torch.nn.Linear(
                in_features=self.config.hidden_size,
                out_features=int(self.config.hidden_size * self.config.mlp_factor),
                dtype=self.config.precision.dtype,
                device=device,
                bias=self.config.bias_terms,
            )
            if self.config.init == "xavier":
                torch.nn.init.xavier_normal_(self.swiglu.weight)
            elif self.config.init == "pythia":
                std = math.sqrt(2 / (5 * self.config.hidden_size))
                torch.nn.init.normal_(self.swiglu.weight, mean=0.0, std=std)
            else:
                assert False

        self.dense_out = torch.nn.Linear(
            in_features=int(self.config.hidden_size * self.config.mlp_factor),
            out_features=self.config.hidden_size,
            dtype=self.config.precision.dtype,
            device=device,
            bias=self.config.bias_terms,
        )

        if self.config.init == "xavier":
            torch.nn.init.xavier_normal_(self.dense_in.weight)
            torch.nn.init.xavier_normal_(self.dense_out.weight)
        elif self.config.init == "pythia":
            std = math.sqrt(2 / (5 * self.config.hidden_size))
            torch.nn.init.normal_(self.dense_in.weight, mean=0.0, std=std)
            std = 2 / self.config.num_layers / math.sqrt(self.config.hidden_size)
            torch.nn.init.normal_(self.dense_out.weight, mean=0.0, std=std)
        else:
            assert self.config.init == "normal"
            torch.nn.init.normal_(
                self.dense_in.weight,
                mean=0.0,
                std=self.config.init_std_global_gain
                / math.sqrt(self.config.hidden_size),
            )
            torch.nn.init.normal_(
                self.dense_out.weight,
                mean=0.0,
                std=self.config.init_std_global_gain
                / math.sqrt(self.config.hidden_size),
            )

    def forward(
        self,
        hidden_states: torch.Tensor,
    ):
        if self.config.mlp_type == "swiglu":
            hidden_states_ = self.dense_in(hidden_states)
            hidden_states_ = torch.nn.functional.silu(hidden_states_)
            hidden_states = hidden_states_ * self.swiglu(hidden_states)
            hidden_states = self.dense_out(hidden_states)
        else:
            assert self.config.mlp_type == "simple_gelu"
            hidden_states = self.dense_in(hidden_states)
            hidden_states = torch.nn.functional.gelu(hidden_states)
            hidden_states = self.dense_out(hidden_states)

        return hidden_states
