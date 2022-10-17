import csv
import json
import os
import random
import shutil
import six
import numpy as np
import torch
from torch.functional import Tensor
from utils.entities import TokenSpan
import logging

CSV_DELIMETER = ";"
logger = logging.getLogger()


def create_directories_file(f):
    d = os.path.dirname(f)

    if d and not os.path.exists(d):
        os.makedirs(d)

    return f


def create_directories_dir(d):
    if d and not os.path.exists(d):
        os.makedirs(d)

    return d


def create_csv(file_path, *column_names):
    if not os.path.exists(file_path):
        with open(file_path, "w", newline="") as csv_file:
            writer = csv.writer(
                csv_file,
                delimiter=CSV_DELIMETER,
                quotechar="|",
                quoting=csv.QUOTE_MINIMAL,
            )

            if column_names:
                writer.writerow(column_names)


def append_csv(file_path, *row):
    if not os.path.exists(file_path):
        raise Exception("The given file doesn't exist")

    with open(file_path, "a", newline="") as csv_file:
        writer = csv.writer(
            csv_file, delimiter=CSV_DELIMETER, quotechar="|", quoting=csv.QUOTE_MINIMAL
        )
        writer.writerow(row)


def append_csv_multiple(file_path, *rows):
    if not os.path.exists(file_path):
        raise Exception("The given file doesn't exist")

    with open(file_path, "a", newline="") as csv_file:
        writer = csv.writer(
            csv_file, delimiter=CSV_DELIMETER, quotechar="|", quoting=csv.QUOTE_MINIMAL
        )
        for row in rows:
            writer.writerow(row)


def read_csv(file_path):
    lines = []
    with open(file_path, "r") as csv_file:
        reader = csv.reader(
            csv_file, delimiter=CSV_DELIMETER, quotechar="|", quoting=csv.QUOTE_MINIMAL
        )
        for row in reader:
            lines.append(row)

    return lines[0], lines[1:]


def copy_python_directory(source, dest, ignore_dirs=None):
    source = source if source.endswith("/") else source + "/"
    for (dir_path, dir_names, file_names) in os.walk(source):
        tail = "/".join(dir_path.split(source)[1:])
        new_dir = os.path.join(dest, tail)

        if ignore_dirs and True in [(ignore_dir in tail) for ignore_dir in ignore_dirs]:
            continue

        create_directories_dir(new_dir)

        for file_name in file_names:
            if file_name.endswith(".py"):
                file_path = os.path.join(dir_path, file_name)
                shutil.copy2(file_path, new_dir)


def print_arguments(args):
    logger.info("-----------  Configuration Arguments -----------")
    for arg, value in sorted(six.iteritems(vars(args))):
        logger.info("%s: %s" % (arg, value))
    logger.info("------------------------------------------------")


def summarize_dict(summary_writer, dic, name):
    table = "Argument|Value\n-|-"

    for k, v in vars(dic).items():
        row = "\n%s|%s" % (k, v)
        table += row
    summary_writer.add_text(name, table)


def set_seed(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # some cudnn methods can be random even after fixing the seed
    # unless you tell it to be deterministic
    torch.backends.cudnn.deterministic = True


def reset_logger(logger):
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    for f in logger.filters[:]:
        logger.removeFilters(f)


def flatten(l):
    return [i for p in l for i in p]


def get_as_list(dic, key):
    if key in dic:
        return [dic[key]]
    else:
        return []


def extend_tensor(tensor, extended_shape, fill=0):
    tensor_shape = tensor.shape

    extended_tensor = torch.zeros(extended_shape, dtype=tensor.dtype).to(tensor.device)
    extended_tensor = extended_tensor.fill_(fill)

    if len(tensor_shape) == 1:
        extended_tensor[: tensor_shape[0]] = tensor
    elif len(tensor_shape) == 2:
        extended_tensor[: tensor_shape[0], : tensor_shape[1]] = tensor
    elif len(tensor_shape) == 3:
        extended_tensor[
            : tensor_shape[0], : tensor_shape[1], : tensor_shape[2]
        ] = tensor
    elif len(tensor_shape) == 4:
        extended_tensor[
            : tensor_shape[0], : tensor_shape[1], : tensor_shape[2], : tensor_shape[3]
        ] = tensor

    return extended_tensor


def padded_stack(tensors, padding=0):
    dim_count = len(tensors[0].shape)

    max_shape = [max([t.shape[d] for t in tensors]) for d in range(dim_count)]
    padded_tensors = []

    for t in tensors:
        e = extend_tensor(t, max_shape, fill=padding)
        padded_tensors.append(e)

    stacked = torch.stack(padded_tensors)
    return stacked


def batch_index(tensor, index, pad=False):
    if tensor.shape[0] != index.shape[0]:
        raise Exception()

    if not pad:
        return torch.stack([tensor[i][index[i]] for i in range(index.shape[0])])
    else:
        return padded_stack([tensor[i][index[i]] for i in range(index.shape[0])])


def padded_nonzero(tensor, padding=0):
    indices = padded_stack(
        [tensor[i].nonzero().view(-1) for i in range(tensor.shape[0])], padding
    )
    return indices


def swap(v1, v2):
    return v2, v1


def get_span_tokens(tokens, span):
    inside = False
    span_tokens = []

    for t in tokens:
        # print(t.index)
        if t.index == span[0]:
            inside = True

        if inside:
            span_tokens.append(t)

        if inside and t.index + 1 == span[1]:
            return TokenSpan(span_tokens)

    return None


def to_device(batch, device):
    converted_batch = dict()
    for key in batch.keys():
        if not isinstance(batch[key], Tensor):
            converted_batch[key] = batch[key]
            continue
        converted_batch[key] = batch[key].to(device)

    return converted_batch


def iof(a, b):
    iof = 0
    if max(a[0], b[0]) < min(a[1], b[1]):
        iof = (min(a[1], b[1]) - max(a[0], b[0])) / (a[1] - a[0])
    return iof


def pooling(sub, sup_mask, pool_type="mean"):
    sup = None
    if len(sub.shape) == len(sup_mask.shape):
        if pool_type == "mean":
            size = (sup_mask == 1).float().sum(-1).unsqueeze(-1) + 1e-30
            m = (sup_mask.unsqueeze(-1) == 1).float()
            sup = m * sub.unsqueeze(1).repeat(1, sup_mask.shape[1], 1, 1)
            sup = sup.sum(dim=2) / size
        if pool_type == "sum":
            m = (sup_mask.unsqueeze(-1) == 1).float()
            sup = m * sub.unsqueeze(1).repeat(1, sup_mask.shape[1], 1, 1)
            sup = sup.sum(dim=2)
        if pool_type == "max":
            m = (sup_mask.unsqueeze(-1) == 0).float() * (-1e30)
            sup = m + sub.unsqueeze(1).repeat(1, sup_mask.shape[1], 1, 1)
            sup = sup.max(dim=2)[0]
            sup[sup == -1e30] = 0
    else:
        if pool_type == "mean":
            size = (sup_mask == 1).float().sum(-1).unsqueeze(-1) + 1e-30
            m = (sup_mask.unsqueeze(-1) == 1).float()
            sup = m * sub
            sup = sup.sum(dim=2) / size
        if pool_type == "sum":
            m = (sup_mask.unsqueeze(-1) == 1).float()
            sup = m * sub
            sup = sup.sum(dim=2)
        if pool_type == "max":
            m = (sup_mask.unsqueeze(-1) == 0).float() * (-1e30)
            sup = m + sub
            sup = sup.max(dim=2)[0]
            sup[sup == -1e30] = 0
    return sup