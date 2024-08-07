from typing import Optional, Tuple, Dict, Any, Union
import torch


class ParameterMeta:
    def __init__(
        self,
        local_shape: Tuple[int, ...],
        layer_index: Optional[int] = None,
        parameter_name: Optional[str] = None,
    ):
        self.local_shape = local_shape
        self.layer_index = layer_index
        self.parameter_name = parameter_name
        if layer_index is not None:
            self.set_layer_index(layer_index=layer_index)

    def __repr__(self):
        return f"ParameterMeta [{self.parameter_name}] layer_index [{self.layer_index}]"

    @property
    def key(self):
        """
        unique identifer within a constant model architecture independent of layout
        """
        return self.key_for_layer(self.layer_index)

    def key_for_layer(self, layer_index: str):
        return f"layer_index_{layer_index}_parameter_name_{self.parameter_name}"

    def possible_keys(self):
        return [self.key]

    def state_dict(self) -> Dict[str, Any]:
        return {
            "local_shape": self.local_shape,
            "layer_index": self.layer_index,
            "parameter_name": self.parameter_name,
        }

    @classmethod
    def from_state_dict(cls, state_dict: Dict[str, Any]):
        return cls(**state_dict)

    def set_layer_index(self, layer_index: int):
        self.layer_index = layer_index

    def set_parameter_name(self, parameter_name: str):
        self.parameter_name = parameter_name

    def set_is_tied(self, is_tied: bool):
        self.is_tied = is_tied

    @staticmethod
    def register_on_parameter(
        parameter: torch.Tensor,
        layer_index: Optional[int] = None,
        parameter_name: Optional[str] = None,
    ):
        assert not hasattr(
            parameter, "aleph_alpha_scaling_parameter_meta"
        ), "aleph_alpha_scaling_parameter_meta already registered"

        local_shape = tuple(parameter.shape)

        meta = ParameterMeta(
            local_shape=local_shape,
            layer_index=layer_index,
            parameter_name=parameter_name,
        )

        parameter.parameter_meta = meta  # type: ignore

        return meta

    def __eq__(self, o):
        if not isinstance(o, ParameterMeta):
            return False

        return self.key == o.key
