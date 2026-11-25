"""Text trace-file parsers used by io.load_dataset."""
import logging
import re
import numpy as np
import pandas as pd

def parse_tuple_data():
    """Compiles regex for tuple parsing."""
    return re.compile(r'^\s*\(\s*([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\s*,\s*([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\s*,\s*([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\s*\)\s*$')


def _parse_tuple(s: str, tuple_re):
    """Parses a string to a numeric tuple."""
    if not isinstance(s, str):
        return np.nan
    m = tuple_re.match(s.strip())
    if m:
        try:
            return tuple(float(x) for x in m.groups())
        except (ValueError, IndexError):
            return np.nan
    return np.nan


def get_valid_trace_files(traces_dir):
    """Returns valid files for processing."""
    valid_extensions = {'.txt', '.csv', '.shp', '.xyz', '.dat'}
    all_files = list(traces_dir.glob('*.*'))
    valid_files = [file for file in all_files if file.suffix.lower() in valid_extensions]
    return sorted(valid_files)


def process_text_file(tf, tuple_re):
    """Processes trace text files in tabular or raw tuple format."""
    df = read_text_file_manual(tf)

    if df.empty:
        return df

    df["X0"] = df["X0"].apply(lambda x: _parse_tuple(x, tuple_re))
    df["X1"] = df["X1"].apply(lambda x: _parse_tuple(x, tuple_re))

    return df.dropna(subset=["X0", "X1"])


def _extract_two_tuples_from_line(line):
    """Extract the first and last 3D tuples from one trace row."""
    pattern = r'\(\s*[+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?\s*,\s*[+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?\s*,\s*[+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?\s*\)'
    matches = re.findall(pattern, line)
    return [matches[0], matches[-1]] if len(matches) >= 2 else None


def read_text_file_manual(tf):
    """Reads trace text files with or without header."""
    rows = []

    try:
        with open(tf, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()

                if not line or line.startswith("#"):
                    continue

                tuples = _extract_two_tuples_from_line(line)

                if tuples is not None:
                    rows.append(tuples)
                    continue

                parts = re.split(r"\t|\s{2,}", line)

                if len(parts) >= 2:
                    rows.append(parts[:2])

    except Exception as e:
        logging.error(f"Manual reading failed for {tf.name}: {e}")
        return pd.DataFrame()

    return pd.DataFrame(rows, columns=["X0", "X1"])


