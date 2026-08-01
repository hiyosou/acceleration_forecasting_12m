import sys

from tqdm import tqdm


def progress_bar(iterable=None, *, enabled=True, mininterval=1.0, **kwargs):
    return tqdm(
        iterable, disable=not enabled, file=sys.stderr,
        mininterval=mininterval, dynamic_ncols=True, **kwargs,
    )


def progress_message(message, *, enabled=True):
    if enabled:
        tqdm.write(str(message), file=sys.stderr)

