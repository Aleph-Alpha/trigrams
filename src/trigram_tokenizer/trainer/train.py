from typing import Optional
from datetime import timedelta

try:
    from determined.core._context import Context as DeterminedContext  # type: ignore
    from determined.profiler import ProfilerAgent as DeterminedProfilerAgent  # type: ignore
except ImportError:
    print("WARNING: determined not installed, skipping")
    DeterminedContext = None  # type: ignore
    DeterminedProfilerAgent = None  # type: ignore

import torch

from trigram_tokenizer.logging import logger
from trigram_tokenizer.trainer.config import TrainerConfig
from trigram_tokenizer.trainer.context import TrainerContext
from trigram_tokenizer.trainer.trainer import Trainer
from trigram_tokenizer.transformer import TransformerLMHeadModel
from trigram_tokenizer.data import TextDataset
from trigram_tokenizer.optimizer import (
    AdamWOptimizer,
    AdamWParameterGroup,
    AdamWOptimizerParamGroupConfig,
)


def get_parameter_groups(context: TrainerContext, model: TransformerLMHeadModel):
    named_parameters_with_meta_weight_decay = list()
    named_parameters_with_meta_no_weight_decay = list()

    for n, p, m in AdamWOptimizer.named_parameters_with_meta(model):
        if n.endswith(".bias"):
            named_parameters_with_meta_no_weight_decay.append((n, p, m))
        else:
            named_parameters_with_meta_weight_decay.append((n, p, m))

    parameter_counts = [
        len(named_parameters_with_meta_weight_decay),
        len(named_parameters_with_meta_no_weight_decay),
    ]

    parameter_count_total = sum(parameter_counts)
    parameter_count_total_tensor = torch.tensor(
        [parameter_count_total], dtype=torch.long, device="cuda"
    )
    torch.distributed.all_reduce(parameter_count_total_tensor)
    parameter_count_total = int(parameter_count_total_tensor.item())
    assert (
        parameter_count_total > 0
    ), f"did not specifiy any trainable paramters on any rank"

    parameter_groups = []

    parameter_set = set(
        [p[0] for p in named_parameters_with_meta_weight_decay]
        + [p[0] for p in named_parameters_with_meta_no_weight_decay]
    )
    logger.warning(f"training parameters: {parameter_set}")

    parameter_counts_max_tensor = torch.tensor(
        parameter_counts,
        dtype=torch.int,
        device="cuda",
    )
    # collect whether there is at least one non-empty group for weight_decay, resp. no_weight decay parameters on some rank
    torch.distributed.all_reduce(
        parameter_counts_max_tensor, op=torch.distributed.ReduceOp.MAX
    )

    # if at least one rank has a non empty group we need to add the group everywhere since it hangs otherwise
    if parameter_counts_max_tensor[0].item() > 0:
        parameter_groups.append(
            AdamWParameterGroup(
                named_parameters_with_meta=named_parameters_with_meta_weight_decay,
                config=AdamWOptimizerParamGroupConfig(
                    name="weight_decay_params",
                    weight_decay=context.config.training.weight_decay,
                    learning_rate_scheduler=context.config.learning_rate_scheduler,
                ),
            )
        )
    if parameter_counts_max_tensor[1].item() > 0:
        parameter_groups.append(
            AdamWParameterGroup(
                named_parameters_with_meta=named_parameters_with_meta_no_weight_decay,
                config=AdamWOptimizerParamGroupConfig(
                    name="no_weight_decay_params",
                    weight_decay=0.0,
                    learning_rate_scheduler=context.config.learning_rate_scheduler,
                ),
            )
        )

    # Safety check whether the number of optimizer groups is the same on all ranks

    len_param_groups_tensor_list = [
        torch.zeros([1], dtype=torch.int, device="cuda")
        for _ in range(torch.distributed.get_world_size())
    ]
    torch.distributed.all_gather(
        len_param_groups_tensor_list,
        torch.tensor([len(parameter_groups)], dtype=torch.int, device="cuda"),
    )

    len_param_groups_list = [t.item() for t in len_param_groups_tensor_list]
    assert (
        len(set(len_param_groups_list)) == 1
    ), f"Got different number of optimizer groups on different ranks \n {len_param_groups_list}"

    assert len(parameter_groups) > 0, "Number of optimizer groups is zero"

    return parameter_groups


def broadcast_model(transformer: TransformerLMHeadModel):
    """
    broadcast model weights from data parallel rank 0 to all other data parallel ranks
    This ensures the same initial parameter values
    """

    for parameter in transformer.parameters():
        if torch.is_tensor(parameter):
            torch.distributed.broadcast(
                parameter,
                0,
            )


def train_main(
    conf_file: str,
    overwrite_config: Optional[dict] = None,
    determined_context: Optional[DeterminedContext] = None,
    determined_profiler: Optional[DeterminedProfilerAgent] = None,
    master_addr: Optional[str] = None,
    master_port: Optional[str] = None,
    world_size: Optional[str] = None,
    global_rank: Optional[str] = None,
    local_slot: Optional[str] = None,
    return_metrics: bool = False,
):
    if conf_file is None:
        assert overwrite_config is not None
        config = TrainerConfig.from_dict(overwrite_config)
    else:
        config = TrainerConfig.from_yaml(conf_file, overwrite_values=overwrite_config)

    logger.info(
        f"initialize_distributed using master {master_addr}:{master_port} with world_size {config.training.world_size} for rank {global_rank}"
    )
    assert world_size is not None
    assert global_rank is not None
    assert master_addr is not None
    assert master_port is not None
    assert local_slot is not None
    torch.distributed.init_process_group(
        backend="nccl",
        world_size=int(world_size),
        rank=int(global_rank),
        init_method=f"tcp://{master_addr}:{master_port}",
        timeout=timedelta(minutes=20),
    )
    device = f"cuda:{local_slot}"
    torch.cuda.set_device(device)

    logger.configure(
        config=config.logger,
        name=f"RANK {global_rank}",
        global_rank=int(global_rank),
        use_determined=determined_context is not None,
        determined_context=determined_context,
        determined_profiler=determined_profiler,
    )
    logger.log_config(config=config)

    logger.info(f" SEQ {config.architecture.sequence_length}")
    logger.info(f" VOCAB {config.architecture.vocab_size}")

    assert config.architecture.vocab_size == config.tokenizer.vocab_size
    assert config.architecture.sequence_length == config.tokenizer.sequence_length

    context = TrainerContext(config=config)
    transformer = TransformerLMHeadModel(config=config.architecture, device=device)
    broadcast_model(transformer=transformer)
    parameter_count = sum([p.numel() for p in transformer.parameters()])
    logger.log_config_dict(
        {
            "parameter_count": parameter_count,
        }
    )
    parameter_groups = get_parameter_groups(context=context, model=transformer)
    optimizer = AdamWOptimizer(
        config=context.config.optimizer,
        parameter_groups=parameter_groups,
    )
    dataset = TextDataset(config=config.data, tokenizer_config=config.tokenizer)

    trainer = Trainer(
        context=context,
        transformer=transformer,
        optimizer=optimizer,
        dataset=dataset,
        device=device,
    )
    trainer.run_training()
