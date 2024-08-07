from trigram_tokenizer import determined as local_determined_overwrite
from typing import Dict, Optional, List, Union
from trigram_tokenizer.determined.utils import (
    determined_profiler_from_ctx,
    maybe_periodic_stacktraces,
)
from determined import ClusterInfo
from determined.core._context import Context as DeterminedContext  # type: ignore
from determined.profiler import ProfilerAgent as DeterminedProfilerAgent  # type: ignore
import argparse
import os


try:
    import determined as det  # type: ignore
    from determined.core._context import Context as DeterminedContext  # type: ignore
    from determined.profiler import ProfilerAgent as DeterminedProfilerAgent  # type: ignore
except ImportError:
    print("WARNING: determined not installed, skipping")
    DeterminedContext = None  # type: ignore
    DeterminedProfilerAgent = None  # type: ignore

from trigram_tokenizer.trainer.config import TrainerConfig
from trigram_tokenizer.trainer.train import train_main


def main(
    determined_context: DeterminedContext,
    profiler: Optional[DeterminedProfilerAgent],
    overwrite_config: Optional[dict] = None,
    return_metrics: bool = False,
    det_experiment_id: Optional[int] = None,
    det_trial_id: Optional[int] = None,
    info: Optional[ClusterInfo] = None,
) -> Optional[List[Dict[str, Union[float, int]]]]:
    """
    Collects determined launcher arguments and calls training script
    """

    master_addr = os.environ["MASTER_ADDR"]
    master_port = os.environ["MASTER_PORT"]
    world_size = os.environ["WORLD_SIZE"]
    global_rank = os.environ["RANK"]
    local_slot = os.environ[
        "LOCAL_RANK"
    ]  # Torch distributed launcher set name as LOCAL_RANK

    parser = argparse.ArgumentParser(description="process launch")

    # Optional arguments for the launch helper
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="path to config file",
    )
    parser.add_argument("remaining_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    if overwrite_config is None:
        overwrite_config = dict()
    if "training" not in overwrite_config:
        overwrite_config["training"] = dict()
    overwrite_config["training"]["determined_experiment_id"] = det_experiment_id
    overwrite_config["training"]["determined_trial_id"] = det_trial_id
    overwrite_config["training"]["world_size"] = world_size

    try:
        assert info is not None
        hparams = info.trial.hparams
    except:
        hparams = {}
    if "vocab_size" in hparams:
        overwrite_config["tokenizer"] = dict()
        overwrite_config["architecture"] = dict()
        overwrite_config["tokenizer"]["vocab_size"] = hparams["vocab_size"]
        overwrite_config["architecture"]["vocab_size"] = hparams["vocab_size"]
        print(f" >> overwritten vocab_size {hparams['vocab_size']}")
    if "vocab_population" in hparams:
        if "tokenizer" not in overwrite_config:
            overwrite_config["tokenizer"] = dict()

        overwrite_config["tokenizer"]["vocab_population"] = hparams["vocab_population"]
        print(f" >> overwritten vocab_population {hparams['vocab_population']}")

    if "learning_rate" in hparams:
        overwrite_config["learning_rate_scheduler"] = dict()
        overwrite_config["learning_rate_scheduler"]["learning_rate"] = hparams[
            "learning_rate"
        ]
        overwrite_config["learning_rate_scheduler"]["learning_rate_minimum"] = (
            hparams["learning_rate"] / 10
        )
        print(f" >> overwritten learning rate {hparams['learning_rate']}")

    return train_main(
        args.config,
        overwrite_config=overwrite_config,
        determined_context=determined_context,
        master_addr=master_addr,
        master_port=master_port,
        world_size=world_size,
        global_rank=global_rank,
        local_slot=local_slot,
        return_metrics=return_metrics,
    )


if __name__ == "__main__":
    info = det.get_cluster_info()
    assert info is not None
    config_determined = det.ExperimentConfig(info.trial._config)
    det_experiment_id = info.trial.experiment_id
    det_trial_id = info.trial.trial_id

    distributed = det.core.DistributedContext.from_torch_distributed()
    with maybe_periodic_stacktraces(config_determined.debug_enabled()):
        with local_determined_overwrite.core.init(
            distributed=distributed
        ) as determined_context:
            with determined_profiler_from_ctx(
                determined_context, config_determined, info
            ) as profiler:
                main(
                    determined_context,
                    profiler,
                    det_experiment_id=det_experiment_id,
                    det_trial_id=det_trial_id,
                    info=info,
                )
