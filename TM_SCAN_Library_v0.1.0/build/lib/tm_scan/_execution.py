"""Bounded Joblib execution and one-line progress, without global patches."""
import sys
from time import perf_counter
from joblib import Parallel, delayed, effective_n_jobs, parallel_config


def validate_jobs(n_jobs):
    if isinstance(n_jobs, bool) or not isinstance(n_jobs, int) or n_jobs == 0:
        raise ValueError("n_jobs must be a nonzero integer (-1 uses available CPUs).")
    return n_jobs


class Progress:
    def __init__(self, total, label, enabled=True):
        self.total, self.label, self.enabled = total, label, bool(enabled)
        self.start = perf_counter()
        self.last_update = self.start
        self.display_handle = None
        if self.enabled:
            try:
                from IPython import get_ipython
                if get_ipython() is not None and getattr(get_ipython(), "kernel", None) is not None:
                    from IPython.display import display
                    self.display_handle = display({"text/plain": f"{label}: 0/{total}"},
                                                  raw=True, display_id=True)
            except ImportError:
                pass
        self.update(0)

    def update(self, done):
        if self.enabled:
            now = perf_counter()
            if done not in (0, self.total) and now-self.last_update < .1:
                return
            self.last_update = now
            message = f"{self.label}: {done}/{self.total} | {now-self.start:.1f} s"
            if self.display_handle is not None:
                self.display_handle.update({"text/plain": message}, raw=True)
            else:
                print("\r"+message, end="\n" if done == self.total else "", flush=True, file=sys.stdout)


def run_tasks(function, arguments, *, n_jobs=-1, backend="threading", progress=False, label="Analysis"):
    validate_jobs(n_jobs)
    arguments = list(arguments)
    if not arguments:
        return []
    count = min(effective_n_jobs(n_jobs), len(arguments))
    meter = Progress(len(arguments), label, progress)
    if count == 1:
        iterator = (function(*args) for args in arguments)
        output = []
        for i, value in enumerate(iterator, 1):
            output.append(value)
            meter.update(i)
        return output
    options = dict(backend=backend, n_jobs=count)
    if backend == "loky":
        options["inner_max_num_threads"] = 1
    with parallel_config(**options):
        iterator = Parallel(return_as="generator", batch_size="auto", verbose=0)(
            delayed(function)(*args) for args in arguments)
        output = []
        for i, value in enumerate(iterator, 1):
            output.append(value)
            meter.update(i)
    return output
