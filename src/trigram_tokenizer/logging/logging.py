from typing import Optional, Dict, Any, List
import logging
import copy
import json
import subprocess
import os
import requests  # type: ignore
import socket

from enum import Enum
from pydantic import Field, root_validator
from pathlib import Path
from datetime import datetime
from dateutil.parser import parse  # type: ignore

try:
    from determined.core._context import Context as DeterminedContext  # type: ignore
    from determined.profiler import ProfilerAgent as DeterminedProfilerAgent  # type: ignore
except ImportError:
    print("WARNING: determined not installed, skipping")
    DeterminedContext = None  # type: ignore
    DeterminedProfilerAgent = None  # type: ignore

import wandb

import torch

from ..config import BaseConfig

# On top of the config, we have another variable to check if we *really* want tensorboard enabled.
# The reasoning is that tensorboard is that we already store the data in other sinks so we are
# having the metrics data written 3 times at least. On top of that, the determined implementation
# of tensorboard seems to be writing way too many files, which makes backing up our disk brittle.
# So, in the rare case of needing to have tensorboard (we have wandb and determined metrics), this
# environment variable needs to be correctly set, otherwise we won't save tensorboard metrics.
ENABLE_TENSORBOARD_ENV_VAR = "SCALING_ENABLE_TENSORBOARD"


class LogLevel(Enum):
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class ColorFormatter(logging.Formatter):
    grey = "\x1b[38;20m"
    blue = "\x1b[33;94m"
    green = "\x1b[33;92m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"

    COLORS = {
        logging.DEBUG: grey,
        logging.INFO: grey,
        logging.WARNING: yellow,
        logging.ERROR: red,
        logging.CRITICAL: bold_red,
    }

    def __format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)

    def format(self, record):
        """
        Format the specified record as text.

        The record's attribute dictionary is used as the operand to a
        string formatting operation which yields the returned string.
        Before formatting the dictionary, a couple of preparatory steps
        are carried out. The message attribute of the record is computed
        using LogRecord.getMessage(). If the formatting string uses the
        time (as determined by a call to usesTime(), formatTime() is
        called to format the event time. If there is exception information,
        it is formatted using formatException() and appended to the message.
        """

        record.message = record.getMessage()
        if self.usesTime():
            record.asctime = self.formatTime(record, self.datefmt)
        s = self.formatMessage(record)
        if record.exc_info:
            # Cache the traceback text to avoid converting it multiple times
            # (it's constant anyway)
            if not record.exc_text:
                record.exc_text = self.formatException(record.exc_info)
        if record.exc_text:
            if s[-1:] != "\n":
                s = s + "\n"
            s = s + record.exc_text
        if record.stack_info:
            if s[-1:] != "\n":
                s = s + "\n"
            s = s + self.formatStack(record.stack_info)

        color = self.COLORS.get(record.levelno)
        if color is None:
            return s
        else:
            return f"{color}{s}{self.reset}"


class LoggerConfig(BaseConfig):
    log_level: LogLevel = Field(LogLevel.INFO, description="")

    log_dir: Optional[Path] = Field(None, description="")

    metrics_ranks: Optional[List[int]] = Field(
        None,
        description="define the global ranks of process to write metrics. If the list is ommitted or None only rank 0 will write metrics.",
    )

    use_wandb: bool = Field(False, description="")
    wandb_ranks: Optional[List[int]] = Field(
        None,
        description="define the global ranks of process to write to wandb. If the list is ommitted or None only rank 0 will write to wandb.",
    )

    wandb_host: str = Field("https://api.wandb.ai", description="url of the wandb host")
    wandb_team: str = Field(
        "aleph-alpha", description="Team name for Weights and Biases."
    )
    wandb_project: str = Field("trigram-tokenizer", description="wandb project name")
    wandb_group: str = Field("debug", description="wandb project name")
    wandb_api_key: Optional[str] = Field(
        None,
        description="set wandb api key in order not to perform a wandb login first",
    )

    use_tensorboard: bool = Field(False, description="")
    tensorboard_ranks: Optional[List[int]] = Field(
        None,
        description="define the global ranks of process to write to tensorboard. If the list is ommitted or None only rank 0 will write to tensorboard.",
    )

    determined_metrics_ranks: Optional[List[int]] = Field(
        None,
        description="define the global ranks of process to write metrics to determined. If the list is ommitted or None only rank 0 will write to determined.",
    )

    @root_validator(pre=True)
    def add_dates_to_values(cls, values: Dict[Any, Any]) -> Dict[Any, Any]:
        log_dir = values.get("log_dir")
        if log_dir is not None:
            log_dir = Path(log_dir)
            if not is_date(log_dir.name):
                log_dir = log_dir / datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
                values["log_dir"] = log_dir

        wandb_group = values.get("wandb_group")
        if wandb_group is not None:
            if not is_date(wandb_group.split("-")[-1]):
                wandb_group += "-" + datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
                values["wandb_group"] = wandb_group

        return values


def is_date(string, fuzzy=False):
    """
    Return whether the string can be interpreted as a date.

    :param string: str, string to check for date
    :param fuzzy: bool, ignore unknown tokens in string if True
    """
    try:
        parse(string, fuzzy=fuzzy)
        return True

    except ValueError:
        return False


def command_exists(cmd):
    result = subprocess.Popen(f"type {cmd}", stdout=subprocess.PIPE, shell=True)
    return result.wait() == 0


def git_info():
    git_hash_cmd = "git rev-parse --short HEAD"
    git_branch_cmd = "git rev-parse --abbrev-ref HEAD"
    if command_exists("git"):
        try:
            result = subprocess.check_output(git_hash_cmd, shell=True)
            git_hash = result.decode("utf-8").strip()
            result = subprocess.check_output(git_branch_cmd, shell=True)
            git_branch = result.decode("utf-8").strip()
        except subprocess.CalledProcessError:
            git_hash = "unknown"
            git_branch = "unknown"
    else:
        git_hash = "unknown"
        git_branch = "unknown"

    return {
        "git_hash": git_hash,
        "git_branch": git_branch,
    }


def get_wandb_api_key(config: LoggerConfig):
    """Get Weights and Biases API key from ENV or .netrc file. Otherwise return None"""
    if "WANDB_LOCAL" in os.environ:
        return "LOCAL"

    wandb_token = requests.utils.get_netrc_auth(config.wandb_host)
    if wandb_token is not None:
        return wandb_token[1]

    return None


def init_wandb(config: LoggerConfig, global_rank: Optional[int]):
    # try and get key
    wandb_api_key = config.wandb_api_key or get_wandb_api_key(config=config)
    if wandb_api_key is None:
        return False

    os.environ["WANDB_BASE_URL"] = config.wandb_host
    os.environ["WANDB_API_KEY"] = wandb_api_key
    group_name = config.wandb_group
    name = f"{socket.gethostname()}-{global_rank}" if group_name else None
    success = True
    try:
        wandb.init(
            project=config.wandb_project,
            group=group_name,
            name=name,
            save_code=False,
            force=False,
            entity=config.wandb_team,
        )
    except wandb.UsageError as e:
        success = False

    return success


class Logger:
    def __init__(self):
        self._tensorboard_writer = None
        self._use_tensorboard = False
        self._use_wandb = False
        self._use_determined_metrics = False
        self._write_metrics = True

        self._logger = logging.getLogger(name="trigram-tokenizer")
        self._handler = logging.StreamHandler()
        self._file_handler = None
        self.set_formatter()
        self._logger.addHandler(self._handler)
        self.set_level(LogLevel.INFO)

    def set_level(self, log_level: LogLevel):
        level = logging.DEBUG
        if log_level == log_level.INFO:
            level = logging.INFO
        if log_level == log_level.WARNING:
            level = logging.WARNING
        if log_level == log_level.ERROR:
            level = logging.ERROR
        if log_level == log_level.CRITICAL:
            level = logging.CRITICAL
        self._logger.setLevel(level)
        self._handler.setLevel(level)

    def set_formatter(self, name: Optional[str] = None):
        if name is None:
            formatter = ColorFormatter("[%(asctime)s] [%(levelname)s] %(message)s")
        else:
            formatter = ColorFormatter(
                f"[%(asctime)s] [%(levelname)s] [{name}] %(message)s"
            )
        self._handler.setFormatter(formatter)
        if self._file_handler is not None:
            self._file_handler.setFormatter(formatter)

    def configure(
        self,
        config: LoggerConfig,
        name: Optional[str] = None,
        global_rank: Optional[int] = None,
        use_determined: bool = False,
        determined_context: Optional[DeterminedContext] = None,
        determined_profiler: Optional[DeterminedProfilerAgent] = None,
    ):
        self.set_level(config.log_level)
        self._use_determined = use_determined
        self.determined_context = determined_context
        self.determined_profiler = determined_profiler

        if config.log_dir is not None:
            # configure for distributed logging to files
            config.log_dir.mkdir(exist_ok=True, parents=True)
            self._file_handler = logging.FileHandler(
                filename=str(config.log_dir.absolute() / f"log_{name}.log")
            )
            self._logger.addHandler(self._file_handler)

            # configure tensorboard
            if (
                bool(os.getenv(ENABLE_TENSORBOARD_ENV_VAR, False))
                and config.use_tensorboard
                and global_rank is not None
                and (
                    (config.tensorboard_ranks is None and global_rank == 0)
                    or (
                        config.tensorboard_ranks is not None
                        and global_rank in config.tensorboard_ranks
                    )
                )
            ):
                self._use_tensorboard = True
                if self._use_determined:
                    from determined.tensorboard.metric_writers.pytorch import TorchWriter  # type: ignore

                    wrapped_writer = TorchWriter()
                    self._tensorboard_writer = wrapped_writer.writer
                else:
                    from torch.utils.tensorboard import SummaryWriter

                    self._tensorboard_writer = SummaryWriter(
                        log_dir=str(config.log_dir / "tensorboard")
                    )

        # configure wandb
        if (
            config.use_wandb
            and global_rank is not None
            and (
                (config.wandb_ranks is None and global_rank == 0)
                or (
                    config.wandb_ranks is not None and global_rank in config.wandb_ranks
                )
            )
        ):
            self._use_wandb = init_wandb(config=config, global_rank=global_rank)

        # configure determined metrics
        if (
            self._use_determined
            and global_rank is not None
            and (
                (config.determined_metrics_ranks is None and global_rank == 0)
                or (
                    config.determined_metrics_ranks is not None
                    and global_rank in config.determined_metrics_ranks
                )
            )
        ):
            self._use_determined_metrics = True

        # Write metrics for rank 0 if metrics ranks not set or if rank is included in metrics ranks
        if global_rank is not None and (
            (config.metrics_ranks is None and global_rank == 0)
            or (
                config.metrics_ranks is not None and global_rank in config.metrics_ranks
            )
        ):
            self._write_metrics = True
        else:
            self._write_metrics = False

        self.set_formatter(name=name)

    def debug(self, msg: object):
        self._logger.debug(msg=msg)

    def info(self, msg: object):
        self._logger.info(msg=msg)

    def warning(self, msg: object):
        self._logger.warning(msg=msg)

    def error(self, msg: object):
        self._logger.error(msg=msg)

    def critical(self, msg: object):
        self._logger.critical(msg=msg)

    def report_memory(self, name):
        """Simple GPU memory report."""
        mega_bytes = 1024.0 * 1024.0
        string = name + " memory (MB)"
        string += " | allocated: {}".format(torch.cuda.memory_allocated() / mega_bytes)
        string += " | max allocated: {}".format(
            torch.cuda.max_memory_allocated() / mega_bytes
        )
        string += " | reserved: {}".format(torch.cuda.memory_reserved() / mega_bytes)
        string += " | max reserved: {}".format(
            torch.cuda.max_memory_reserved() / mega_bytes
        )
        self.info(string)

    def log_metrics(self, metrics: Dict[str, Any], step: int):
        if self._write_metrics:
            self.info(json.dumps(metrics))

        # report training metrics to determined
        if self._use_determined_metrics:
            determined_metrics = dict(
                {
                    k: (v.item() if isinstance(v, torch.Tensor) else v)
                    for k, v in metrics.items()
                }
            )
            assert self.determined_context is not None

            determined_evaluation_metrics: Dict[str, Any] = {
                k: v
                for k, v in determined_metrics.items()
                if k.startswith("evaluation")
            }
            determined_training_metrics: Dict[str, Any] = {
                k: v
                for k, v in determined_metrics.items()
                if not k.startswith("evaluation")
            }

            self.determined_context.train.report_training_metrics(
                steps_completed=step, metrics=determined_training_metrics
            )

            self.determined_context.train.report_validation_metrics(
                steps_completed=step, metrics=determined_evaluation_metrics
            )

        if self._use_wandb:
            wandb.log(metrics, step=step)

        if self._use_tensorboard and self._tensorboard_writer is not None:
            for k, v in metrics.items():
                self._tensorboard_writer.add_scalar(k, v, step)
            self._tensorboard_writer.flush()

    def log_config(self, config: BaseConfig):
        config_dict = copy.deepcopy(config.as_dict())
        self.log_config_dict(config_dict=config_dict)

    def log_config_dict(self, config_dict: Dict):
        if self._use_wandb:
            wandb.config.update(config_dict, allow_val_change=True)

        if self._use_tensorboard and self._tensorboard_writer is not None:
            for name, value in config_dict.items():
                self._tensorboard_writer.add_text(name, str(value))


logger = Logger()
