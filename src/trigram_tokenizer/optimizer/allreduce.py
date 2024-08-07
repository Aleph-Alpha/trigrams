from typing import List
import torch


def split_dtypes(tensors):
    supported_types = [
        "torch.cuda.HalfTensor",
        "torch.cuda.FloatTensor",
        "torch.cuda.DoubleTensor",
        "torch.cuda.BFloat16Tensor",
    ]

    for t in tensors:
        assert (
            t.type() in supported_types
        ), f"attempting to reduce an unsupported grad type: {t.type()}"

    buckets = []
    for i, dtype in enumerate(supported_types):
        bucket = [t for t in tensors if t.type() == dtype]
        if bucket:
            buckets.append(bucket)
    return buckets


def allreduce_bucket(bucket: List[torch.Tensor]):
    tensor = torch._C._nn.flatten_dense_tensors(bucket)

    tensor_to_allreduce = tensor

    if tensor.dtype == torch.bfloat16:
        # always all-reduce in fp32 precision
        communication_data_type = torch.float32
        tensor_to_allreduce = tensor.to(communication_data_type)
    else:
        communication_data_type = tensor_to_allreduce.dtype

    tensor_to_allreduce.div_(torch.distributed.get_world_size())

    torch.distributed.all_reduce(tensor_to_allreduce)

    if communication_data_type != tensor.dtype and tensor is not tensor_to_allreduce:
        tensor.copy_(tensor_to_allreduce)

    return tensor


def allreduce_and_copy(bucket: List[torch.Tensor]):
    stream = torch.cuda.current_stream()

    with torch.cuda.stream(stream):
        allreduced = allreduce_bucket(
            bucket=bucket,
        )
        for buf, synced in zip(
            bucket,
            torch._C._nn.unflatten_dense_tensors(allreduced, bucket),
        ):
            buf.copy_(synced)


def allreduce_no_retain(
    bucket: List[torch.Tensor],
    numel_per_bucket=500000000,
):
    small_bucket = []
    numel = 0
    for tensor in bucket:
        small_bucket.append(tensor)
        numel = numel + tensor.numel()
        if numel > numel_per_bucket:
            allreduce_and_copy(
                bucket=small_bucket,
            )
            small_bucket = []
            numel = 0

    if len(small_bucket) > 0:
        allreduce_and_copy(
            bucket=small_bucket,
        )


def buffered_allreduce(
    bucket: List[torch.Tensor],
    numel_per_bucket=500000000,
):
    split_buckets = split_dtypes(bucket)

    for b in split_buckets:
        allreduce_no_retain(
            bucket=b,
            numel_per_bucket=numel_per_bucket,
        )
