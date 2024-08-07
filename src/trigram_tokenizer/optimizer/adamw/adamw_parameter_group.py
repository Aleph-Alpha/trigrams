from typing import List, Iterable, Tuple, Union, Optional
from pydantic import Field

import torch

from trigram_tokenizer.logging import logger

from ..base import BaseConfig

from ..learning_rate_scheduler import (
    LearningRateSchedulerConfig,
    LearningRateScheduler,
    LearningRateDecayStyle,
)
from ..parameter_meta import ParameterMeta


# align nccl all-gather send buffers to 4-byte boundary
# 4-byte alignment/sizeof(fp16) = 2
NCCL_START_ALIGNMENT_FACTOR = 2


def flatten_dense_tensors_aligned(
    tensor_list: List[torch.Tensor], alignment: int
) -> Tuple[torch.Tensor, int]:
    """
    create a flat tensor aligned at the alignment boundary
    """
    num_elements = sum(t.numel() for t in tensor_list)
    remaining = num_elements % alignment

    if remaining:
        zero_padding = alignment - remaining
        pad_tensor = torch.zeros(
            zero_padding,
            device="cpu",
            dtype=tensor_list[0].dtype,
        )
        padded_tensor_list = tensor_list + [pad_tensor]
    else:
        padded_tensor_list = tensor_list
        zero_padding = 0

    return torch._C._nn.flatten_dense_tensors(padded_tensor_list), zero_padding


def get_data_parallel_partitions(tensor: torch.Tensor):
    """
    views the tensor as multiple partitions and returns
    those partitions
    """
    partitions = []

    total_num_elements = tensor.numel()

    base_size = total_num_elements // torch.distributed.get_world_size()
    remaining = total_num_elements % torch.distributed.get_world_size()

    start = 0
    for id in range(torch.distributed.get_world_size()):
        partition_size = base_size
        if id < remaining:
            partition_size = partition_size + 1
        partitions.append(tensor.narrow(0, start, partition_size))
        start = start + partition_size

    # verify that data partition start locations are 4-byte aligned
    for partitioned_data in partitions:
        assert partitioned_data.data_ptr() % (2 * NCCL_START_ALIGNMENT_FACTOR) == 0

    # verify that data partition start locations are 4-byte aligned
    for partitioned_data in partitions:
        assert partitioned_data.data_ptr() % (2 * NCCL_START_ALIGNMENT_FACTOR) == 0

    return partitions


class AdamWOptimizerParamGroupConfig(BaseConfig):
    name: Optional[str] = Field(
        None,
        description="Name of the parameter group for logging",
    )

    learning_rate_scheduler: LearningRateSchedulerConfig = Field(
        LearningRateSchedulerConfig(
            learning_rate=0.0001,
            learning_rate_minimum=0.0,
            learning_rate_decay_style=LearningRateDecayStyle.COSINE,
            learning_rate_decay_iters=50000,
            learning_rate_warmup_steps=2000,
        ),
        description="Configuration of the parameter group's learning rate schedule",
    )

    weight_decay: float = Field(
        1e-2,
        description="Weight decay for all parameters within the parameter group",
    )


class AdamWParameterGroup:
    """
    Parameter group to which the same train behavior is applied
    Different parameter groups are initialized for different values of
    weight decay and learning rate schedule.
    """

    def __init__(
        self,
        named_parameters_with_meta: List[Tuple[str, torch.Tensor, ParameterMeta]],
        config: AdamWOptimizerParamGroupConfig,
    ):
        # record config
        self.config = config

        # instantiate a learning rate scheduler for the parameter group
        self.learning_rate_scheduler = LearningRateScheduler(
            self.config.learning_rate_scheduler
        )

        self.parameter_names: List[str] = list()
        self.parameter_metas: List[ParameterMeta] = list()
        self.parameters_original: List[torch.Tensor] = list()
        self.parameters_optimized: List[torch.Tensor] = list()

        # zero parameters
        self.zero_dtype: Optional[torch.dtype] = None
        self.zero: bool = False
        self.zero_padding: int = 0
        self.parameters_original_flat: Optional[torch.Tensor] = None
        self.parameters_original_partitions: Optional[List[torch.Tensor]] = None
        self.float_parameter_partition_owned: Optional[torch.Tensor] = None

        # record parameters
        # we can only iterate once through named parameters
        # this is potentially a generator that can be consumed
        self.dummy_parameters = list()
        if len(named_parameters_with_meta) == 0:
            dummy_parameter = torch.tensor([0.0], dtype=torch.bfloat16).cuda()
            dummy_meta = ParameterMeta.register_on_parameter(
                dummy_parameter,
                layer_index=-1,
                parameter_name="dummy_parameter",
            )
            self.dummy_parameters.append(dummy_parameter)
            named_parameters_with_meta = [
                (
                    "dummy_parameter",
                    dummy_parameter,
                    dummy_meta,
                )
            ]

        for n, p, m in named_parameters_with_meta:
            self.parameter_names.append(n)
            self.parameter_metas.append(m)
            self.parameters_original.append(p)

    def initialize(self, zero: bool):
        self.zero = zero

        # prepare to be optimized parameters
        if zero:
            # assert that all parameters have the same datatype for zero
            self.zero_dtype = self.parameters_original[0].dtype
            assert all(
                [p.dtype == self.zero_dtype for p in self.parameters_original]
            ), f"all zero optimizer parameters in a single group must have the same data type. different data types may be splitted in different optimizer groups"
            assert all(
                [p.device.type == "cuda" for p in self.parameters_original]
            ), f"all zero optimizer parameters in a single group must have be on cuda."

            self.move_to_cpu(self.parameters_original)
            (
                parameters_original_flat,
                self.zero_padding,
            ) = flatten_dense_tensors_aligned(
                self.parameters_original,
                NCCL_START_ALIGNMENT_FACTOR * torch.distributed.get_world_size(),
            )
            self.parameters_original_flat = parameters_original_flat.cuda()
            self._update_original_parameters()
            self.parameters_original_partitions = get_data_parallel_partitions(
                self.parameters_original_flat
            )
            assert self.parameters_original_partitions is not None

            # a single 32-bit partition of the parallel partitioned parameters
            # that this process will update
            self.float_parameter_partition_owned = (
                self.parameters_original_partitions[torch.distributed.get_rank()]
                .to(torch.cuda.current_device())  # type: ignore
                .clone()
                .float()
                .detach()
            )
            assert self.float_parameter_partition_owned is not None
            self.parameters_optimized = [self.float_parameter_partition_owned]
            (
                self.parameters_owned,
                self.parameters_not_owned,
                self.parameter_owned_offset,
            ) = self._get_partition_info(
                parameters_original=self.parameters_original,
                partition_size=len(self.float_parameter_partition_owned),
            )

            self.parameter_coordinates = self._get_parameter_coordinates(
                parameters_original=self.parameters_original,
                partition_size=len(self.float_parameter_partition_owned),
            )

            # assert that all parameters still have the same datatype
            # this was a bug before
            assert all(
                [p.dtype == self.zero_dtype for p in self.parameters_original]
            ), f"all zero optimizer parameters in a single group must have the same data type. different data types may be splitted in different optimizer groups"
            assert all(
                [p.device.type == "cuda" for p in self.parameters_original]
            ), f"all zero optimizer parameters in a single group must have be on cuda."

        else:
            # collect float parameters
            # optimization will happen on full precision
            float_parameters = list()
            for p in self.parameters_original:
                if p.dtype == torch.float32:
                    float_parameter = p
                else:
                    float_parameter = p.clone().float().detach()
                    float_parameter.parameter_meta = (  # type: ignore
                        p.parameter_meta  # type: ignore
                    )
                float_parameters.append(float_parameter)
            self.parameters_optimized = float_parameters

        # in case the internal optimizer needs it
        for p in self.parameters_optimized:
            p.requires_grad = True

        # create a parameter dict that is suitable for adamw
        # this dict will be handed to adamw and can be manipulated
        self.parameter_dict = {
            "params": self.parameters_optimized,
            "weight_decay": self.config.weight_decay,
            "lr": self.config.learning_rate_scheduler.learning_rate,
        }

    @staticmethod
    def move_to_cpu(tensor_list):
        for tensor in tensor_list:
            tensor.data = tensor.data.cpu()

    def _update_original_parameters(self):
        """
        set model bit16 weight to slices of flattened buffer
        """
        updated_params = torch._C._nn.unflatten_dense_tensors(
            self.parameters_original_flat,
            self.parameters_original,
        )
        for original, updated in zip(self.parameters_original, updated_params):
            original.data = updated.data

    @staticmethod
    def _get_parameter_coordinates(
        parameters_original: List[torch.Tensor], partition_size: int
    ):
        parameter_coordinates = list()

        current_index = 0
        for parameter_original in parameters_original:
            parameter_original_size = parameter_original.numel()

            # record the parameters coordinates
            parameter_coordinate = dict()
            for data_parallel_rank in range(torch.distributed.get_world_size()):
                data_parallel_partition_start_index = (
                    partition_size * data_parallel_rank
                )
                data_parallel_partition_end_index = partition_size * (
                    data_parallel_rank + 1
                )

                if (
                    data_parallel_partition_start_index <= current_index
                    and current_index < data_parallel_partition_end_index
                ):
                    # the start of the current tensor is within the start and end of the data parallel partition
                    start = current_index - data_parallel_partition_start_index
                    end = min(
                        current_index
                        + parameter_original_size
                        - data_parallel_partition_start_index,
                        data_parallel_partition_end_index,
                    )
                    offset = 0
                elif (
                    current_index < data_parallel_partition_start_index
                    and data_parallel_partition_start_index
                    < (current_index + parameter_original_size)
                ):
                    # the start of the tensor is before the start of the current partition, but some part of the tensor is still in the current partition
                    start = 0
                    end = min(
                        current_index
                        + parameter_original_size
                        - data_parallel_partition_start_index,
                        data_parallel_partition_end_index,
                    )
                    offset = data_parallel_partition_start_index - current_index
                else:
                    start = None
                    end = None
                    offset = None

                # start in data parallel slice
                # end in data parallel slice
                # offset of data parallel part of parameter within parameter
                parameter_coordinate[data_parallel_rank] = (
                    start,
                    end,
                    offset,
                    parameter_original.shape,
                )

            # remember coordinates
            parameter_coordinates.append(parameter_coordinate)

            # increment current index for next parameter
            current_index = current_index + parameter_original_size

        return parameter_coordinates

    @staticmethod
    def _get_partition_info(
        parameters_original: List[torch.Tensor], partition_size: int
    ):
        parameters_owned = []
        parameters_not_owned = []

        start_index = partition_size * torch.distributed.get_rank()
        end_index = partition_size * (torch.distributed.get_rank() + 1)

        current_index = 0  # index pointing to a current position in a flattened tensor
        parameter_owned_offset = 0

        for parameter_original in parameters_original:
            parameter_original_size = parameter_original.numel()

            if start_index <= current_index and current_index < end_index:
                # the start of the current tensor is within the start and end of the data parallel partition
                parameters_owned.append(parameter_original)
            elif current_index < start_index and start_index < (
                current_index + parameter_original_size
            ):
                # the start of the tensor is before the start of the current partition, but some part of the tensor is still in the current partition
                parameters_owned.append(parameter_original)
                assert (
                    parameter_owned_offset == 0
                ), "This can happen either zero or only once as this must be the first tensor in the partition"
                parameter_owned_offset = start_index - current_index
            else:
                parameters_not_owned.append(parameter_original)

            current_index = current_index + parameter_original_size

        return parameters_owned, parameters_not_owned, parameter_owned_offset

    def set_dummy_grad(self):
        for parameter in self.dummy_parameters:
            parameter.grad = torch.zeros_like(parameter.data)

    def step_prequel(self, step_index: int):
        """ """

        # Set the learning rate for `step_index`
        self.parameter_dict["lr"] = self.learning_rate_scheduler.get_lr(
            step_index=step_index
        )

        # Copying gradients to fp32 to work with fp32 parameters
        if self.zero:
            # if zero is used we will need to find the gradients applicable to the current data parallel slice

            # first remove the grads that are not owned
            for parameter in self.parameters_not_owned:
                parameter.grad = None

            assert self.float_parameter_partition_owned is not None
            partition_size = len(self.float_parameter_partition_owned)
            current_start = 0
            self.float_parameter_partition_owned.grad = torch.zeros_like(
                self.float_parameter_partition_owned
            )
            for parameter_index, parameter in enumerate(self.parameters_owned):
                # make sure there is a grad
                if parameter.grad is None:
                    parameter_grad = torch.zeros_like(parameter)
                else:
                    parameter_grad = parameter.grad.contiguous()

                # compute offset in case the current dp slice starts in the middle of the tensor
                parameter_offset = 0
                parameter_elements = parameter_grad.numel()
                if parameter_index == 0 and self.parameter_owned_offset > 0:
                    parameter_offset = self.parameter_owned_offset
                    parameter_elements = (
                        parameter_elements - self.parameter_owned_offset
                    )

                # cut parameter in case it ends at the next dp slice
                if parameter_elements > (partition_size - current_start):
                    parameter_elements = partition_size - current_start

                # we need a narrow view of the tensor based on the tensor offset and number of elements that
                # we need from this tensor
                if parameter_offset > 0 or parameter_elements < parameter_grad.numel():
                    self.float_parameter_partition_owned.grad[
                        current_start : current_start + parameter_elements
                    ].copy_(
                        parameter_grad.view(-1).narrow(
                            0, int(parameter_offset), int(parameter_elements)
                        )
                    )
                else:
                    self.float_parameter_partition_owned.grad[
                        current_start : current_start + parameter_elements
                    ].copy_(parameter_grad.view(-1))

                current_start = current_start + parameter_elements
                parameter.grad = None  # Grad on the parameter is not needed any more
        else:
            # if zero is not used the optimized parameters retain the shape (or are even only pointers to same memory in case of fp32)
            for optimized, original in zip(
                self.parameters_optimized, self.parameters_original
            ):
                if id(original) == id(optimized):
                    continue

                if original.grad is None:
                    optimized.grad = torch.zeros(
                        optimized.size(),
                        dtype=optimized.dtype,
                        device=optimized.device,
                    )
                else:
                    optimized.grad = original.grad.to(optimized.dtype)
                    original.grad = None  # Grad on the parameter is not needed any more

    def get_original_grads(self):
        result = list()

        for original in self.parameters_original:
            assert (
                original.grad is not None
            ), f"parameter {original.parameter_meta.parameter_name} has grad None"
            result.append(original.grad.data)
        return result

    def get_optimized_grads(self):
        return [optimized.grad.data for optimized in self.parameters_optimized]

    def step_sequel(self, numel_per_bucket: int):
        """ """
        if self.zero:
            # local grads are not needed any more
            assert self.float_parameter_partition_owned is not None
            self.float_parameter_partition_owned.grad = None

            # get updated parameter values from optimizer
            assert self.parameters_original_partitions is not None
            self.parameters_original_partitions[
                torch.distributed.get_rank()
            ].data.copy_(self.float_parameter_partition_owned.data)

            # collect to local instances of flattened tensors
            num_shards = max(
                1,
                self.float_parameter_partition_owned.numel()
                * torch.distributed.get_world_size()
                // numel_per_bucket,
            )

            shard_size = self.float_parameter_partition_owned.numel() // num_shards

            # Enforce nccl/rccl alignment of start location of each shard
            shard_size = shard_size - (shard_size % NCCL_START_ALIGNMENT_FACTOR)

            num_elements = shard_size

            assert (
                shard_size * num_shards <= self.float_parameter_partition_owned.numel()
            )

            for shard_id in range(num_shards):
                if shard_id == (num_shards - 1):
                    num_elements = (
                        self.float_parameter_partition_owned.numel()
                        - shard_id * shard_size
                    )

                shard_list = []
                for dp_id in range(torch.distributed.get_world_size()):
                    curr_shard = (
                        self.parameters_original_partitions[dp_id]
                        .narrow(0, shard_id * shard_size, num_elements)
                        .detach()
                    )
                    shard_list.append(curr_shard)
                torch.distributed.all_gather(
                    shard_list,
                    shard_list[torch.distributed.get_rank()],
                )

            # update local parameters
            self._update_original_parameters()

        else:
            # Copying updated parameters to original parameters
            for optimized, original in zip(
                self.parameters_optimized, self.parameters_original
            ):
                if id(original) == id(optimized):
                    continue

                # copy data from fp32 to fp16
                original.data.copy_(optimized.data)

    def scale_grads_and_get_grad_norm_squared(
        self,
        optimized_grads: bool,
        loss_scale: float = 1.0,
        asam_norm: bool = False,
    ):
        grad_norm_squared = 0.0

        parameters_for_norm = (
            self.parameters_optimized if optimized_grads else self.parameters_original
        )
        for param in parameters_for_norm:
            # accumulate grad norms
            parameter_meta: ParameterMeta = param.parameter_meta  # type: ignore

            assert (
                param.grad is not None
            ), f"parameter {parameter_meta.parameter_name} has no grad"

            if asam_norm:
                param.grad.data.mul_(torch.abs(param) / loss_scale)
            else:
                param.grad.data.mul_(1 / loss_scale)

            if param.grad.dtype == torch.float16:
                grad_norm_squared += param.grad.float().norm(2) ** 2
            else:
                grad_norm_squared += param.grad.norm(2) ** 2

        if (
            grad_norm_squared == float("inf")
            or grad_norm_squared == -float("inf")
            or grad_norm_squared != grad_norm_squared
        ):
            raise RuntimeError(f"grad norm is {grad_norm_squared}")

        return grad_norm_squared

    def scale_grads(self, scale_factor: float):
        """
        scale the gradients on full precision copy
        """

        for optimized in self.parameters_optimized:
            if (
                optimized.grad is not None
            ):  # this should not happen, but the type is technicall an optional tensor
                optimized.grad.data.mul_(scale_factor)

    def log_state(self):
        """
        Log useful information for debugging and overal information
        """
        logger.debug(f"lr {self.parameter_dict['lr']}")

    def get_learning_rate(self):
        """
        get the currently set learning rate
        """
        return self.parameter_dict["lr"]

    def state_dict(self):
        """
        Get a state_dict representing the state of the parameter group
        """

        if self.zero:
            parameters_optimized = [None for _ in self.parameters_original]
            parameters_optimized_flat = self.float_parameter_partition_owned
            parameter_coordinates = self.parameter_coordinates
        else:
            parameters_optimized = self.parameters_optimized
            parameters_optimized_flat = None
            parameter_coordinates = [None for _ in parameters_optimized]

            assert len(parameters_optimized) == len(parameter_coordinates)

        return {
            "parameter_names": self.parameter_names,
            "parameter_metas": [m.state_dict() for m in self.parameter_metas],
            "parameters_optimized": parameters_optimized,
            "parameters_optimized_flat": parameters_optimized_flat,
            "parameter_coordinates": parameter_coordinates,
        }

    def load_state_dict(self, state_dict, zero_load_static: bool = False):
        """
        Load the state into an already initialized parameter group
        """

        assert (
            self.parameter_names == state_dict["parameter_names"]
        ), "parameters changed"
        assert [m.state_dict() for m in self.parameter_metas] == state_dict[
            "parameter_metas"
        ], "parameter metas changed"

        if zero_load_static:
            assert self.float_parameter_partition_owned is not None
            self.float_parameter_partition_owned.data.copy_(
                state_dict["parameters_optimized_flat"]
            )
            return

        assert len(self.parameters_optimized) == len(state_dict["parameters_optimized"])
        for name, current, parameter_state_dict in zip(
            self.parameter_names,
            self.parameters_optimized,
            state_dict["parameters_optimized"],
        ):
            assert (
                current.shape == parameter_state_dict.shape
            ), f"shape of parameter {name} changed; now {current.shape} from checkpoint {parameter_state_dict.shape}"
            current.data.copy_(parameter_state_dict.data)

    @property
    def trainable_parameters_original(self):
        for p in self.parameters_original:
            if p is None:
                continue
            if p.grad is None:
                continue

            yield p

    @property
    def trainable_parameters_optimized(self):
        for p in self.parameters_optimized:
            if p is None:
                continue
            if p.grad is None:
                continue

            yield p

    def zero_grad(self, set_to_none=True):
        # original parameters
        for p in self.trainable_parameters_original:
            if set_to_none:
                p.grad = None
            elif p.grad is not None:
                p.grad.detach_()
                p.grad.zero_()

        # optimized parameters
        for p in self.trainable_parameters_optimized:
            if set_to_none:
                p.grad = None
            elif p.grad is not None:
                p.grad.detach_()
                p.grad.zero_()

    def refresh_optimized_params(self):
        """
        Update the optimized parameter copies from the original copies
        the optimizer runs only on fp32 parameters
        """
        if self.zero:
            # if zero is used we will need to map the parameter values to the current data parallel slice
            # parameters_original_partitions is a pointer
            assert self.parameters_original_partitions is not None
            assert self.float_parameter_partition_owned is not None
            self.float_parameter_partition_owned.data.copy_(
                self.parameters_original_partitions[torch.distributed.get_rank()].data
            )

        else:
            for optimized, original in zip(
                self.parameters_optimized, self.parameters_original
            ):
                if original is not None:
                    optimized.data.copy_(original.data)

    @property
    def parameters_for_overflow_check(self) -> List[torch.Tensor]:
        return self.parameters_original
