from typing import List, Dict, Union, Tuple
import collections
import math
import copy
from pydantic import Field
from pathlib import Path


import torch

from trigram_tokenizer.logging import logger

from ..base import BaseOptimizer, BaseConfig, OptimizerStepOutput
from .loss_scaler import LossScaler, LossScalerConfig

from ..allreduce import allreduce_no_retain
from .adamw_parameter_group import (
    AdamWParameterGroup,
    flatten_dense_tensors_aligned,
    NCCL_START_ALIGNMENT_FACTOR,
    get_data_parallel_partitions,
)
from ..parameter_meta import ParameterMeta


class AdamWOptimizerConfig(BaseConfig):
    beta1: float = Field(
        0.9,
        description="First coefficient used for computing running averages of gradient and its square",
    )

    beta2: float = Field(
        0.95,
        description="Second coefficient used for computing running averages of gradient and its square",
    )

    eps: float = Field(
        1e-8,
        description="term added to the denominator to improve numerical stability (default: 1e-8)",
    )

    gradient_clipping: float = Field(
        0.0, description="clip global l2 grads to this value, deactivate if 0.0", ge=0.0
    )

    allreduce_bucket_size: int = Field(
        500000000, description="number of floating points to allreduce in one go", gt=0
    )

    loss_scaler: LossScalerConfig = Field(
        LossScalerConfig(
            enable=False,
            initial_scale=2.0**32,
            window=1000,
            hysteresis=2,
            consecutive_hysteresis=False,
            min_scale=1.0,
            factor=2.0,
        ),
        description="Configuration of the loss scaler",
    )

    zero: bool = Field(False, description="enable zero stage 1 optimizer")

    zero_save_static: bool = Field(
        False,
        description="Save zero state dict without merging parameters and optimizer states. This may be used in large scale trainings to save and load checkpoints faster and not run oom.",
    )

    debug_log: bool = Field(False)


class AdamWOptimizer(BaseOptimizer):
    def __init__(
        self,
        config: AdamWOptimizerConfig,
        parameter_groups: List[AdamWParameterGroup],
    ):
        """
        Wrapper around AdamW Optimizer taking care of parallelization
        """
        self.config = config
        self.parameter_groups = parameter_groups

        self._assert_no_parameter_duplicates()

        for parameter_group in self.parameter_groups:
            parameter_group.initialize(zero=self.config.zero)

        self.optimizer = torch.optim.AdamW(
            [p.parameter_dict for p in self.parameter_groups],
            betas=(self.config.beta1, self.config.beta2),
            eps=self.config.eps,
        )

        # initialize all parameter groups with step
        self.step_index: int = 0

        # loss scaler
        self.loss_scaler = LossScaler(
            config=config.loss_scaler, parameter_groups=self.parameter_groups
        )

    @staticmethod
    def named_parameters_with_meta(
        model: torch.nn.Module,
    ):
        named_parameters_with_meta = list()
        for parameter_name, parameter in model.named_parameters():
            if not hasattr(parameter, "parameter_meta"):
                ParameterMeta.register_on_parameter(
                    parameter=parameter,
                    layer_index=0,
                    parameter_name=parameter_name,
                )
            meta: ParameterMeta = parameter.parameter_meta  # type: ignore
            named_parameters_with_meta.append((parameter_name, parameter, meta))
        return named_parameters_with_meta

    def _assert_no_parameter_duplicates(self):
        counter = collections.Counter()
        id_to_name = dict()
        for parameter_group in self.parameter_groups:
            for name, parameter in zip(
                parameter_group.parameter_names,
                parameter_group.parameters_original,
            ):
                counter[id(parameter)] += 1
                id_to_name[id(parameter)] = name

        parameters_occuring_more_than_once = [
            id_to_name[id] for id, count in counter.items() if count > 1
        ]
        assert (
            len(parameters_occuring_more_than_once) == 0
        ), f"parameters occuring more than once: {parameters_occuring_more_than_once}"

    def zero_grad(self, set_to_none: bool = True):
        """
        Zero half precision parameter grads.
        """
        for parameter_group in self.parameter_groups:
            parameter_group.zero_grad(set_to_none=set_to_none)

    def backward(self, loss: torch.Tensor):
        """
        execute backward pass on loss and potentially add loss scaling
        """
        # scale for gradient accumulation steps
        loss = loss.float()
        loss = self.loss_scaler.scale_loss(loss=loss)
        loss.backward()

    def step(self):
        """
        Do a model step, this optimizes model parameters and zeros gradients
        """

        # remember the last step taken
        self.step_index += 1

        # give dummy parameters a grad
        for parameter_group in self.parameter_groups:
            parameter_group.set_dummy_grad()

        # apply loss scaler and potentially skip step
        loss_scaler_output = self.loss_scaler.step()
        if loss_scaler_output.overflow is not None and loss_scaler_output.overflow:
            logger.warning(f"loss scaler encountered overflow, skipping step")
            self.zero_grad(set_to_none=True)
            return OptimizerStepOutput(
                global_grad_norm=None,
                global_grad_norm_clipped=None,
                learning_rates=None,
                overflow=loss_scaler_output.overflow,
                no_overflow_steps=loss_scaler_output.no_overflow_steps,
                current_loss_scale=loss_scaler_output.current_loss_scale,
                debug_dict=None,
            )

        # Allreduce gradients over dp copies
        # this assumes that the prequel moved gradients to the full precision to be optimized copies
        # in order not to create redundant copies in full precision, zero reduces grads on the original parameters
        # communication will happen in full precision in case of bfloat
        if self.config.zero:
            self.allreduce_gradients(optimized_grads=False)
            global_grad_norm = self.scale_grads_and_get_global_grad_norm(
                optimized_grads=False
            )

        if self.config.debug_log:
            assert self.config.zero
            debug_dict = dict()
            for param_group in self.parameter_groups:
                for param in param_group.parameters_original:
                    param_name = f"{param.parameter_meta.parameter_name}-layer-{param.parameter_meta.layer_index}"
                    if param.grad is not None:
                        debug_dict[f"debug/{param_name}-grad-norm"] = float(
                            param.grad.norm().item()
                        )
                    debug_dict[f"debug/{param_name}-norm"] = float(param.norm().item())
                    debug_dict[f"debug/{param_name}-pct-zero"] = 1.0 - float(
                        (param.count_nonzero() / param.numel()).item()
                    )

        else:
            debug_dict = None

        # Step in learning rate scheduler
        # This will be setting the learning rate for the current step and
        # move gradients to full precision parameter copies
        # for zero this moves gradients the local copies
        for param_group in self.parameter_groups:
            param_group.step_prequel(step_index=self.step_index)

        # Allreduce gradients over dp copies
        # this assumes that the prequel moved gradients to the full precision to be optimized copies
        if not self.config.zero:
            self.allreduce_gradients(optimized_grads=True)
            global_grad_norm = self.scale_grads_and_get_global_grad_norm(
                optimized_grads=True
            )

        # Clip gradients
        was_clipped = self.clip_gradients(global_grad_norm=global_grad_norm)
        global_grad_norm_clipped = (
            None  # cannot be done with gradients being released immediatelly
        )

        # step in actual parameters
        self.optimizer.step()

        # This will move optimized values to half precision parameter copies
        for param_group in self.parameter_groups:
            param_group.step_sequel(
                numel_per_bucket=self.config.allreduce_bucket_size,
            )

        # FP32 grad should never exist outside of the step function
        # For speed, set model fp16 grad to None by default
        self.zero_grad(set_to_none=True)

        # collect learning rates
        learning_rates = dict()
        for param_group_index, param_group in enumerate(self.parameter_groups):
            name = param_group.config.name
            if name is None:
                name = f"parameter_group_{param_group_index}"
            learning_rates[name] = param_group.get_learning_rate()

        return OptimizerStepOutput(
            global_grad_norm=global_grad_norm,
            global_grad_norm_clipped=global_grad_norm_clipped,
            learning_rates=learning_rates,
            overflow=loss_scaler_output.overflow,
            no_overflow_steps=loss_scaler_output.no_overflow_steps,
            current_loss_scale=loss_scaler_output.current_loss_scale,
            debug_dict=debug_dict,
        )

    def scale_grads_and_get_global_grad_norm(
        self, optimized_grads: bool, asam_norm: bool = False
    ):
        # aggregate to global grad norm for all groups
        # this assumes that the param_group_grad_norm_squared aggregates over the full data parallel copy

        loss_scale = 1.0
        if self.loss_scaler.config.enable:
            loss_scale = self.loss_scaler._current_scale

        param_group_grad_norms_squared = list()
        for param_group in self.parameter_groups:
            param_group_grad_norm_squared = (
                param_group.scale_grads_and_get_grad_norm_squared(
                    optimized_grads=optimized_grads,
                    loss_scale=loss_scale,
                    asam_norm=asam_norm,
                )
            )
            param_group_grad_norms_squared.append(param_group_grad_norm_squared)
        global_grad_norm = math.sqrt(sum(param_group_grad_norms_squared))

        return global_grad_norm

    def allreduce_gradients(self, optimized_grads: bool):
        if torch.distributed.get_world_size() == 1:
            return

        gradients = list()
        for parameter_group in self.parameter_groups:
            if optimized_grads:
                gradients.extend(parameter_group.get_optimized_grads())
            else:
                gradients.extend(parameter_group.get_original_grads())

        allreduce_no_retain(
            bucket=gradients,
            numel_per_bucket=self.config.allreduce_bucket_size,
        )

    def clip_gradients(self, global_grad_norm: float):
        # Do not execute if no gradient clipping is defined
        if self.config.gradient_clipping == 0.0:
            return False

        # Do not execute if the global grad norm is small enough
        if global_grad_norm < self.config.gradient_clipping:
            return False

        # actually clip the grads
        scale_factor = self.config.gradient_clipping / global_grad_norm

        # if self.loss_scaler.config.enable:
        #     scale_factor *= self.loss_scaler._current_scale
        for param_group in self.parameter_groups:
            param_group.scale_grads(scale_factor)

        return True

    def refresh_optimizer_after_model_change(self):
        """
        Update the full precision parameter copies from the half precision copies
        the optimizer runs only on fp32 parameters
        """
        for parameter_group in self.parameter_groups:
            parameter_group.refresh_optimized_params()

    def log_state(self):
        """
        Log useful information for debugging and overal information
        """
        for param_group in self.parameter_groups:
            param_group.log_state()

    def state_dict(self):
        """
        Get a state_dict fully representing the optimizer state
        A load of such state dict fully restores the state of the optimizer.
        """
        return {
            "step_index": self.step_index,
            "parameter_groups": [pg.state_dict() for pg in self.parameter_groups],
            "optimizer": self.optimizer.state_dict(),
            "loss_scaler": self.loss_scaler.state_dict(),
        }

    def _get_ordered_parameter_metas_and_layers(self):
        """
        collect parameter meta data
        assumptions are
          - torch.optim.AdamW honors the parameter order
          - we can use the param index
          - parameter order and index is the same across model parallel ranks

        """
        parameter_metas: List[ParameterMeta] = list()
        layer_indices = set()
        for parameter_group in self.parameter_groups:
            for parameter in parameter_group.parameters_original:
                parameter_meta: ParameterMeta = parameter.parameter_meta  # type: ignore
                parameter_metas.append(parameter_meta)
                layer_indices.add(parameter_meta.layer_index)

        return parameter_metas, layer_indices

    def save_checkpoint(self, dir: Union[Path, str]):
        """
        Save the optimizer state to a directory.
        Assumption is that there are no name collisions of files.
        """

        if torch.distributed.get_rank() != 0 and not self.config.zero:
            return

        logger.info(f"starting optimizer checkpoint save")

        # load metadata
        parameter_metas, layer_indices = self._get_ordered_parameter_metas_and_layers()

        # get local state dict
        state_dict_local = self.state_dict()

        if self.config.zero and self.config.zero_save_static:
            dir = Path(dir)
            torch.save(
                state_dict_local,
                str(
                    dir / f"optimizer_state_static_dp_{torch.distributed.get_rank()}.pt"
                ),
            )
            logger.info(f"saved static optimizer checkpoint")
            return

        # initialize merged state dict and copy components that are constant for model parallel
        # one merged state dict is initialized for each model layer
        # duplicated states are saved to all layer indices because we might load only one of them later

        for layer_index in layer_indices:
            state_dict_for_layer: Dict = dict()
            state_dict_for_layer = dict()
            state_dict_for_layer["step_index"] = state_dict_local["step_index"]
            state_dict_for_layer["loss_scaler"] = state_dict_local["loss_scaler"]
            state_dict_for_layer["parameters"] = dict()
            state_dict_for_layer["optimizer_param_groups"] = list()

            # initialize to be saved parameter groups for all layers
            for parameter_group_local, optimizer_parameter_group_local in zip(
                state_dict_local["parameter_groups"],
                state_dict_local["optimizer"]["param_groups"],
            ):
                optimizer_param_group = copy.deepcopy(optimizer_parameter_group_local)
                optimizer_param_group[
                    "params"
                ] = list()  # this will not be valid in the checkpoint
                state_dict_for_layer["optimizer_param_groups"].append(
                    optimizer_param_group
                )

            # save all parameters to state dict
            for (
                step,
                parameter_meta_local,
                parameter_full_local,
                parameter_exp_avg_local,
                parameter_exp_avg_sq_local,
            ) in self.iterate_parameters_for_saving(
                state_dict_local,
                parameter_metas,
                optimizer_parameter_group_local,
                layer_index,
            ):
                # merge the parameter
                merged_parameter_full = parameter_full_local.cpu()

                # create metadata for merged parameter
                parameter_meta = ParameterMeta(
                    local_shape=tuple(merged_parameter_full.shape),
                    layer_index=parameter_meta_local.layer_index,
                    parameter_name=parameter_meta_local.parameter_name,
                )
                assert (
                    parameter_meta_local.key == parameter_meta.key
                ), "key changed after merge"

                # collect the parameters optimizer state
                parameter_optimizer_state = dict()
                parameter_optimizer_state["step"] = step
                parameter_optimizer_state["exp_avg"] = parameter_exp_avg_local.cpu()
                parameter_optimizer_state[
                    "exp_avg_sq"
                ] = parameter_exp_avg_sq_local.cpu()

                # record parameter in state dict
                assert parameter_meta.layer_index is not None
                all_layer_indices = set([parameter_meta.layer_index])
                if layer_index in all_layer_indices:
                    state_dict_for_layer["parameters"][parameter_meta.key] = {
                        "parameter": merged_parameter_full,
                        "meta": parameter_meta.state_dict(),
                        "optimizer_state": parameter_optimizer_state,
                    }

            # collect optimizer states and merge parameters
            # this changes the structure of the optimizer state dict by breaking the reference to the local count

            if torch.distributed.get_rank() == 0:
                dir = Path(dir)
                torch.save(
                    state_dict_for_layer,
                    str(dir / f"optimizer_state_layer_{layer_index}.pt"),
                )

        logger.info(f"saved optimizer checkpoint")

    def iterate_parameters_for_saving(
        self,
        state_dict_local,
        parameter_metas,
        optimizer_parameter_group_local,
        layer_index,
    ):
        global_parameter_index = -1
        for parameter_group_local, optimizer_parameter_group_local in zip(
            state_dict_local["parameter_groups"],
            state_dict_local["optimizer"]["param_groups"],
        ):
            for (
                parameter_name_local,
                parameter_meta_local,
                parameter_full_local,
                parameter_coordinates,
            ) in zip(
                parameter_group_local["parameter_names"],
                parameter_group_local["parameter_metas"],
                parameter_group_local["parameters_optimized"],
                parameter_group_local["parameter_coordinates"],
            ):
                # increment global parameter count as used in adam
                global_parameter_index += 1
                if not self.config.zero:
                    assert (
                        global_parameter_index
                        in optimizer_parameter_group_local["params"]
                    )

                # reinitialize parameter meta from state dict
                parameter_meta_local = ParameterMeta.from_state_dict(
                    parameter_meta_local
                )
                all_layer_indices = set([parameter_meta_local.layer_index])
                if layer_index not in all_layer_indices:
                    continue

                assert parameter_metas[global_parameter_index] == parameter_meta_local
                if (
                    parameter_name_local is not None
                    and parameter_meta_local.parameter_name is not None
                ):
                    assert (
                        parameter_name_local == parameter_meta_local.parameter_name
                    ), f"inconsistent parameter naming {parameter_name_local} vs. {parameter_meta_local.parameter_name}"

                local_parameter_optimizer_state = state_dict_local["optimizer"][
                    "state"
                ][
                    (
                        optimizer_parameter_group_local["params"][0]
                        if self.config.zero
                        else global_parameter_index
                    )
                ]
                step = local_parameter_optimizer_state["step"]  # constant

                if self.config.zero:
                    # collect parameter from data parallel copies

                    parameters_optimized_flat = parameter_group_local[
                        "parameters_optimized_flat"
                    ]

                    start, end, offset, shape = parameter_coordinates[
                        torch.distributed.get_rank()
                    ]
                    assert (
                        parameters_optimized_flat.dtype == torch.float32
                    ), "this assert is for paranoia reasons, optimized parameters should always be in float32"
                    parameter_comms = torch.zeros(
                        (3, torch.prod(torch.tensor(list(shape), dtype=torch.long))),
                        dtype=parameters_optimized_flat.dtype,
                        device=parameters_optimized_flat.device,
                    )
                    if all([start is not None, end is not None, offset is not None]):
                        local_full = parameters_optimized_flat[start:end]
                        parameter_comms.data[
                            0, offset : offset + local_full.numel()
                        ].copy_(local_full.data)

                        local_exp = local_parameter_optimizer_state["exp_avg"][
                            start:end
                        ]
                        parameter_comms.data[
                            1, offset : offset + local_exp.numel()
                        ].copy_(local_exp.data)

                        local_exp_sq = local_parameter_optimizer_state["exp_avg_sq"][
                            start:end
                        ]
                        parameter_comms.data[
                            2, offset : offset + local_exp.numel()
                        ].copy_(local_exp_sq.data)

                    torch.distributed.all_reduce(
                        parameter_comms,
                        op=torch.distributed.ReduceOp.SUM,
                    )

                    parameter_comms = parameter_comms.reshape([3] + list(shape))
                    parameter_full_local = parameter_comms[0]
                    parameter_exp_avg_local = parameter_comms[1]
                    parameter_exp_avg_sq_local = parameter_comms[2]
                else:
                    parameter_exp_avg_local = local_parameter_optimizer_state["exp_avg"]
                    parameter_exp_avg_sq_local = local_parameter_optimizer_state[
                        "exp_avg_sq"
                    ]

                assert (
                    parameter_full_local.shape
                    == parameter_exp_avg_local.shape
                    == parameter_exp_avg_sq_local.shape
                )

                yield step, parameter_meta_local, parameter_full_local, parameter_exp_avg_local, parameter_exp_avg_sq_local

    def load_checkpoint(self, dir: Union[Path, str]):
        """
        Load the state into an already initialized optimizer
        """
        logger.info(f"loading optimizer checkpoint from {dir}")
        if self.config.zero and self.config.zero_save_static:
            dir = Path(dir)
            checkpoint = torch.load(
                str(
                    dir / f"optimizer_state_static_dp_{torch.distributed.get_rank()}.pt"
                ),
            )

            # load constant state
            self.step_index = checkpoint[
                "step_index"
            ]  # constant, does not matter which layer to use
            self.loss_scaler.load_state_dict(
                checkpoint["loss_scaler"]
            )  # constant, does not matter which layer to use
            self.optimizer.load_state_dict(checkpoint["optimizer"])
            for parameter_group_index, (
                parameter_group,
                adamw_parameter_group,
                loaded_state_dict,
            ) in enumerate(
                zip(
                    self.parameter_groups,
                    self.optimizer.param_groups,
                    checkpoint["parameter_groups"],
                )
            ):
                parameter_group.load_state_dict(
                    loaded_state_dict, zero_load_static=True
                )

                # adamw deepcopies
                parameter_group.parameter_dict = adamw_parameter_group

            logger.info(f"loaded static optimizer checkpoint")
            return

        # get currently initialized local parameters
        parameter_metas, layer_indices = self._get_ordered_parameter_metas_and_layers()
        layer_indices = list(layer_indices)

        # load all local layers
        dir = Path(dir)
        state_dict_by_layer = dict()
        for layer_index in layer_indices:
            state_dict_file = dir / f"optimizer_state_layer_{layer_index}.pt"
            state_dict_by_layer[layer_index] = torch.load(str(state_dict_file))

        # make sure all paramters that should be constant are constant
        if len(layer_indices) > 1:
            for layer_index_compare in layer_indices[1:]:
                assert (
                    state_dict_by_layer[layer_indices[0]]["step_index"]
                    == state_dict_by_layer[layer_index_compare]["step_index"]
                )
                assert (
                    state_dict_by_layer[layer_indices[0]]["loss_scaler"]
                    == state_dict_by_layer[layer_index_compare]["loss_scaler"]
                )
                assert (
                    state_dict_by_layer[layer_indices[0]]["optimizer_param_groups"]
                    == state_dict_by_layer[layer_index_compare][
                        "optimizer_param_groups"
                    ]
                )

        # load constant state
        self.step_index = state_dict_by_layer[layer_indices[0]][
            "step_index"
        ]  # constant, does not matter which layer to use
        self.loss_scaler.load_state_dict(
            state_dict_by_layer[layer_indices[0]]["loss_scaler"]
        )  # constant, does not matter which layer to use

        # localize parameters and collect by key
        parameters = dict()
        for _layer_index, state_dict_for_layer in state_dict_by_layer.items():
            for _param_key, parameter_state_dict in state_dict_for_layer[
                "parameters"
            ].items():
                aleph_alpha_parameter_meta = ParameterMeta.from_state_dict(
                    parameter_state_dict["meta"]
                )

                for possible_key in aleph_alpha_parameter_meta.possible_keys():
                    parameters[possible_key] = parameter_state_dict

        # collect and load optimizer state
        optimizer_state_dict: Dict = dict()
        optimizer_state_dict_current = self.optimizer.state_dict()
        optimizer_state_dict["state"] = {}
        optimizer_state_dict["param_groups"] = list()
        assert len(optimizer_state_dict_current["param_groups"]) == len(
            state_dict_by_layer[layer_indices[0]]["optimizer_param_groups"]
        )
        for param_group_current, param_group_loaded in zip(
            optimizer_state_dict_current["param_groups"],
            state_dict_by_layer[layer_indices[0]]["optimizer_param_groups"],
        ):
            # get the pointers to the current parameters
            param_group_loaded["params"] = copy.deepcopy(param_group_current["params"])

            # set the param groups state
            optimizer_state_dict["param_groups"].append(param_group_loaded)

            # set the parameter states
            if self.config.zero:
                for parameter_group_index, parameter_group in enumerate(
                    self.parameter_groups
                ):
                    exp_avg_list = [
                        parameters[parameter_meta.key]["optimizer_state"]["exp_avg"]
                        for parameter_meta in parameter_group.parameter_metas
                    ]
                    exp_avg_tensor, _zero_padding = flatten_dense_tensors_aligned(
                        exp_avg_list,
                        NCCL_START_ALIGNMENT_FACTOR
                        * torch.distributed.get_world_size(),
                    )
                    exp_avg = get_data_parallel_partitions(exp_avg_tensor)[
                        torch.distributed.get_rank()
                    ]

                    exp_avg_sq_list = [
                        parameters[parameter_meta.key]["optimizer_state"]["exp_avg_sq"]
                        for parameter_meta in parameter_group.parameter_metas
                    ]
                    exp_avg_sq_tensor, _zero_padding = flatten_dense_tensors_aligned(
                        exp_avg_sq_list,
                        NCCL_START_ALIGNMENT_FACTOR
                        * torch.distributed.get_world_size(),
                    )
                    exp_avg_sq = get_data_parallel_partitions(exp_avg_sq_tensor)[
                        torch.distributed.get_rank()
                    ]

                    optimizer_state_dict["state"][parameter_group_index] = {
                        "step": parameters[parameter_group.parameter_metas[0].key][
                            "optimizer_state"
                        ]["step"],
                        "exp_avg": exp_avg,
                        "exp_avg_sq": exp_avg_sq,
                    }

            else:
                for param_index in param_group_loaded["params"]:
                    assert (
                        param_index not in optimizer_state_dict["state"]
                    ), "duplicate param index"
                    optimizer_state_dict["state"][param_index] = parameters[
                        parameter_metas[param_index].key
                    ]["optimizer_state"]

        self.optimizer.load_state_dict(optimizer_state_dict)

        for parameter_group_index, (
            parameter_group,
            adamw_parameter_group,
        ) in enumerate(
            zip(
                self.parameter_groups,
                self.optimizer.param_groups,
            )
        ):
            current_state_dict = parameter_group.state_dict()
            loaded_state_dict = dict()
            loaded_state_dict["parameter_names"] = copy.deepcopy(
                current_state_dict["parameter_names"]
            )
            loaded_state_dict["parameter_metas"] = copy.deepcopy(
                current_state_dict["parameter_metas"]
            )

            if self.config.zero:
                parameter_list = [
                    parameters[parameter_meta.key]["parameter"]
                    for parameter_meta in parameter_group.parameter_metas
                ]
                parameter_tensor, _zero_padding = flatten_dense_tensors_aligned(
                    parameter_list,
                    NCCL_START_ALIGNMENT_FACTOR * torch.distributed.get_world_size(),
                )
                parameter = get_data_parallel_partitions(parameter_tensor)[
                    torch.distributed.get_rank()
                ]
                parameters_optimized = [parameter]
            else:
                parameters_optimized = list()
                for parameter_name, parameter_meta_state_dict in zip(
                    current_state_dict["parameter_names"],
                    current_state_dict["parameter_metas"],
                ):
                    aleph_alpha_parameter_meta = ParameterMeta.from_state_dict(
                        parameter_meta_state_dict
                    )
                    assert parameter_name == aleph_alpha_parameter_meta.parameter_name
                    parameters_optimized.append(
                        parameters[aleph_alpha_parameter_meta.key]["parameter"]
                    )
            loaded_state_dict["parameters_optimized"] = parameters_optimized
            parameter_group.load_state_dict(loaded_state_dict)

            # adamw deepcopies
            parameter_group.parameter_dict = adamw_parameter_group

        logger.info(f"loaded optimizer checkpoint")
