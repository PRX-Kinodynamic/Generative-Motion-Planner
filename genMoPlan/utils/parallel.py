import multiprocessing as mp
from functools import partial
from tqdm import tqdm

def _apply_with_args_indexed(func, item):
    index, args = item
    return index, func(*args)

def parallelize(func, args_list, kwargs=None, num_processes=None, show_progress=True, desc=None):
    """
    Parallelize a function call across multiple processes.
    
    Args:
        func: The function to parallelize
        args: List of argument tuples to pass to the function
        kwargs: Dictionary of keyword arguments to pass to all function calls (optional)
        num_processes: Number of processes to use (defaults to cpu_count)
        show_progress: Whether to show a progress bar
    
    Returns:
        List of results from the function calls
    """
    if num_processes is None:
        num_processes = mp.cpu_count()
    
    if kwargs is None:
        kwargs = {}
    
    # Create a wrapper function that includes the kwargs
    if kwargs:
        process_func = partial(func, **kwargs)
    else:
        process_func = func
    
    with mp.Pool(num_processes) as pool:
        if show_progress:
            total_tasks = len(args_list)
            results = [None] * total_tasks
            indexed_args = list(enumerate(args_list))
            iterator = pool.imap_unordered(partial(_apply_with_args_indexed, process_func), indexed_args, chunksize=1)
            for idx, res in tqdm(iterator, total=total_tasks, desc=desc):
                results[idx] = res
        else:
            results = pool.starmap(process_func, args_list)
    
    return results


def parallelize_toggle(func, args_list, kwargs=None, num_processes=None, show_progress=True, desc=None, parallel=True):
    if parallel:
        return parallelize(func, args_list, kwargs, num_processes, show_progress, desc)
    else:
        if show_progress:
            return [func(*args, **kwargs) for args in tqdm(args_list, desc=desc)]
        return [func(*args, **kwargs) for args in args_list]