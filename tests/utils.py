import math
import shutil
import os

from pathlib import Path

import torch


def get_empty_cache_dir(name: str) -> Path:
    cache_dir = Path(__file__).parent.absolute() / "files" / "tmp" / name
    if cache_dir.is_dir():
        shutil.rmtree(str(cache_dir))
    cache_dir.mkdir(exist_ok=True, parents=True)

    return cache_dir


def normal_round(n, digits=0):
    """
    python rounding rounds 0.5 down
    this function fixes it and rounds up
    """
    n = n * (10**digits)
    if n - math.floor(n) < 0.5:
        return float(math.floor(n)) / (10**digits)
    return float(math.ceil(n)) / (10**digits)


def rounded_equal(a: float, b: float, digits=10):
    """
    sometimes float have different numbers of digits due to serialization and rounding
    this function checks if two floats are equal while allowing for one flaot to be rounded
    If different numbers of digits are encountered, the more precise number is rounded
    """
    # no rounding necessary if both are equal
    if a == b:
        return True

    a = float(a)
    b = float(b)

    # find number of characters
    str_a = str(a)
    str_b = str(b)
    digits_a = len(str_a.split(".")[1])
    digits_b = len(str_b.split(".")[1])

    # if same length we know that the two floats must truly be differen
    if digits_a < digits_b:
        round_to_digits = min(digits_a, digits)
    else:
        # b is shorter, a is rounded
        round_to_digits = min(digits_b, digits)

    # if the number itself has been rounded, rounded again at the last position can result in problems
    if min(digits_a, digits_b) - 1 == round_to_digits:
        round_to_digits += 1

    a = normal_round(a, round_to_digits)
    b = normal_round(b, round_to_digits)
    return a == b


def assert_nested_dicts_equal(result, target, message=None, precision=None):
    """
    compare two dictionaries and do readable assertions
    """
    assert isinstance(result, dict), "result is a dict"
    assert isinstance(target, dict), "target is a dict"

    assert set(result.keys()) == set(target.keys()), (
        f"result and target have different keys: {set(result.keys())} vs. {set(target.keys())}"
        + ("" if message is None else " (" + message + ")")
    )

    for k, r_v in result.items():
        t_v = target[k]
        assert type(r_v) == type(t_v), (
            "result and target have different value types for "
            + str(k)
            + (
                f": {type(r_v)} with value {r_v} vs. {type(t_v)} with value {t_v}"
                if message is None
                else " (" + message + f"): {type(r_v)} vs. {type(t_v)}"
            )
        )
        if isinstance(r_v, dict):
            assert_nested_dicts_equal(
                result=r_v,
                target=t_v,
                message=(str(k) if message is None else message + "." + str(k)),
                precision=precision,
            )
        elif isinstance(r_v, list):
            assert_nested_lists_equal(
                result=r_v,
                target=t_v,
                message=(str(k) if message is None else message + "." + str(k)),
                precision=precision,
            )
        else:
            if precision is not None and isinstance(r_v, float):
                assert abs(r_v - t_v) < 0.1**precision, (
                    "result and target have different values for "
                    + str(k)
                    + "; r_v == "
                    + str(r_v)
                    + " ("
                    + str(type(r_v))
                    + "); t_v == "
                    + str(t_v)
                    + "("
                    + str(type(r_v))
                    + ")"
                    + ("" if message is None else " (" + message + ")")
                    + "; precision == "
                    + str(precision)
                )
            else:
                if torch.is_tensor(r_v) and torch.is_tensor(t_v):
                    assert (r_v == t_v).all(), (
                        "result and target have different values for "
                        + str(k)
                        + "; r_v == "
                        + str(r_v)
                        + " ("
                        + str(type(r_v))
                        + "); t_v == "
                        + str(t_v)
                        + "("
                        + str(type(r_v))
                        + ")"
                        + ("" if message is None else " (" + message + ")")
                    )
                else:
                    assert r_v == t_v, (
                        "result and target have different values for "
                        + str(k)
                        + "; r_v == "
                        + str(r_v)
                        + " ("
                        + str(type(r_v))
                        + "); t_v == "
                        + str(t_v)
                        + "("
                        + str(type(r_v))
                        + ")"
                        + ("" if message is None else " (" + message + ")")
                    )


def assert_nested_lists_equal(result, target, message=None, precision=None):
    assert isinstance(result, list), "result is a list"
    assert isinstance(target, list), "target is a list"

    assert len(result) == len(target), "result and target have different lengths" + (
        "" if message is None else " (" + message + ")"
    )
    for i, (r_v, t_v) in enumerate(zip(result, target)):
        assert type(r_v) == type(t_v), (
            "result and target have different value types for list item "
            + str(i)
            + ("" if message is None else " (" + message + ")")
        )
        if isinstance(r_v, dict):
            assert_nested_dicts_equal(
                result=r_v,
                target=t_v,
                message=(
                    "list item " + str(i)
                    if message is None
                    else message + "." + "list item " + str(i)
                ),
                precision=precision,
            )
        elif isinstance(r_v, list):
            assert_nested_lists_equal(
                result=r_v,
                target=t_v,
                message=(
                    "list item " + str(i)
                    if message is None
                    else message + "." + "list item " + str(i)
                ),
                precision=precision,
            )
        else:
            if precision is not None and isinstance(r_v, float):
                assert rounded_equal(r_v, t_v, digits=precision), (
                    "result and target have different values"
                    + "; r_v == "
                    + str(r_v)
                    + "("
                    + str(type(r_v))
                    + ")"
                    + "; t_v == "
                    + str(t_v)
                    + "("
                    + str(type(t_v))
                    + ")"
                    + ("" if message is None else " (" + message + ")")
                    + "; precision == "
                    + str(precision)
                )
            elif torch.is_tensor(r_v) and torch.is_tensor(t_v):
                assert (r_v == t_v).all().item(), (
                    "result and target have different values"
                    + "; r_v == "
                    + str(r_v)
                    + "("
                    + str(type(r_v))
                    + ")"
                    + "; t_v == "
                    + str(t_v)
                    + "("
                    + str(type(t_v))
                    + ")"
                    + ("" if message is None else " (" + message + ")")
                )
            else:
                assert r_v == t_v, (
                    "result and target have different values"
                    + "; r_v == "
                    + str(r_v)
                    + "("
                    + str(type(r_v))
                    + ")"
                    + "; t_v == "
                    + str(t_v)
                    + "("
                    + str(type(t_v))
                    + ")"
                    + ("" if message is None else " (" + message + ")")
                )


def find_free_port():
    """https://stackoverflow.com/questions/1365265/on-localhost-how-do-i-pick-a-free-port-number"""
    import socket
    from contextlib import closing

    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(("", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return str(s.getsockname()[1])


# Worker timeout *after* the first worker has completed.
PROCESS_TIMEOUT = 120


def dist_init(
    run_func,
    master_port,
    local_rank,
    world_size,
    return_dict,
    *func_args,
    **func_kwargs,
):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(master_port)
    os.environ["WORLD_SIZE"] = str(world_size)
    # NOTE: unit tests don't support multi-node so local_rank == global rank
    os.environ["RANK"] = str(local_rank)
    os.environ["LOCAL_SLOT"] = str(local_rank)

    run_func(return_dict=return_dict, *func_args, **func_kwargs)


def dist_launcher(run_func, world_size, master_port, *func_args, **func_kwargs):
    """Launch processes and gracefully handle failures."""
    ctx = torch.multiprocessing.get_context("spawn")
    manager = ctx.Manager()
    return_dict = manager.dict()
    # Spawn all workers on subprocesses.
    processes = []
    for local_rank in range(world_size):
        p = ctx.Process(
            target=dist_init,
            args=(
                run_func,
                master_port,
                local_rank,
                world_size,
                return_dict,
                *func_args,
            ),
            kwargs=func_kwargs,
        )
        p.start()
        processes.append(p)

    # Now loop and wait for a test to complete. The spin-wait here isn't a big
    # deal because the number of processes will be O(#GPUs) << O(#CPUs).
    any_done = False
    any_failed = False
    while not any_done:
        for p in processes:
            if not p.is_alive():
                any_done = True
            if p.exitcode is not None:
                any_failed = any_failed or (p.exitcode != 0)

    if any_failed:
        for p in processes:
            # If it the process hasn't terminated, kill it because it hung.
            if p.is_alive():
                p.terminate()
            if p.is_alive():
                p.kill()

    # Wait for all other processes to complete
    for p in processes:
        p.join(PROCESS_TIMEOUT)

    # Collect exitcodes and terminate hanging process
    failures = []
    for rank, p in enumerate(processes):
        if p.exitcode is None:
            # If it still hasn't terminated, kill it because it hung.
            p.terminate()
            if p.is_alive():
                p.kill()
            failures.append(f"Worker {rank} hung.")
        elif p.exitcode < 0:
            failures.append(f"Worker {rank} killed by signal {-p.exitcode}")
        elif p.exitcode > 0:
            failures.append(f"Worker {rank} exited with code {p.exitcode}")

    if len(failures) > 0:
        raise RuntimeError("\n".join(failures))

    return dict(return_dict)
