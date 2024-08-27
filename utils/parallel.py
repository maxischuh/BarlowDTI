import multiprocessing
import numpy as np
import psutil
from typing import *


def parallel(function: Callable, n_jobs: int, x: List, *args) -> List:
    """Higher order function to run other functions on multiple processes

    Simple parallelization utility, slices the input list x in chunks and
    executes the function on each chunk in different processes. Not suited
    for functions that have already multithreading/processing implemented.

    Args:
        function:   callable to run on different processes
        n_jobs:     how many cores to use
        x:          list (M,) to use as input for function
        *args:      optional arguments for function

    Returns:
        Object (M,) containing the output of function. Content and type depend
        on function. If function returns list, then parallel will also return
        a list. If function returns a numpy array, then parallel will return an
        array.
    """

    # check that parallelization is required. n_jobs might be passed as 1 by
    # i.e. Dataset methods if they notice that the loaded HTS is too large
    # to be used on different cores.
    if n_jobs > 1:
        # split list in chunks
        chunks = split_list(x, n_jobs)

        # create list of tuples containing the chunks and *args
        args = stitch_args(chunks, args)

        # create multiprocessing pool and run function on chunks
        pool = multiprocessing.Pool(n_jobs)
        output = pool.starmap(function, args)
        pool.close()

        # unroll output (list of function outputs) into a single object
        # of size M
        if isinstance(output[0], list):
            unrolled = [x for k in output for x in k]
        elif isinstance(output[0], np.ndarray):
            unrolled = np.concatenate(output, axis=0)

    else:
        # run function normally
        unrolled = function(x, *args)

    return unrolled


def stitch_args(chunks: List[List], args: Tuple) -> List[Tuple]:
    """
    Stitches together the chunks to be run in parallel and optional function
    arguments into tuples
    """
    output = [[x] for x in chunks]
    for i in range(len(output)):
        for j in range(len(args)):
            output[i].append(args[j])

    return [tuple(x) for x in output]


def split_list(x: List, n_jobs: int) -> List[List]:
    """
    Converts a list into a list of lists of size n_jobs.
    """
    idxs = np.array_split(range(len(x)), n_jobs)
    output = [0] * n_jobs
    for i in range(n_jobs):
        output[i] = [x[k] for k in idxs[i]]

    return output
