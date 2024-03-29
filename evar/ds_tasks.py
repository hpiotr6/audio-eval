"""Downstream task definitions."""

from evar.common import (pd, np, WORK, METADATA_DIR)


_defs = {
    # folds, unit_sec, weighted loss, data_folder (None if task name is the folder name), balanced training when fine-tining
    'us8k': [10, 4.0, False, None, False],
    'esc50': [5, 5.0, False, None, False],
    'fsd50k': [1, 7.6358, False, None, False], ## Changed to NOT balanced: to make it the same as PaSST.
    'fsdnoisy18k': [1, 8.25, False, None, False],
    'gtzan': [1, 30.0, False, None, False],
    'nsynth': [1, 4.0, False, None, False],
    'cremad': [1, 2.5, False, None, False],
    'spcv1': [1, 1.0, False, None, False],
    'spcv2': [1, 1.0, False, None, False],
    'surge': [1, 4.0, False, None, False],
    'vc1': [1, 8.2, False, None, False],
    'voxforge': [1, 5.8, False, None, False],
    'as20k': [1, 10.0, False, 'as', False],
    'as': [1, 10.0, False, 'as', True],
}

_fs_table = {
    16000: '16k',
    22000: '22k', # Following COALA that uses 22,000 Hz
    32000: '32k',
    44100: '44k',
    48000: '48k',
}


def get_defs(cfg, task):
    """Get task definition parameters.

    Returns:
        pathname (str): Metadata .csv file path.
        wav_folder (str): "work/16k/us8k" for example.
        folds (int): Number of LOOCV folds or 1. 1 means no cross validation.
        unit_sec (float): Unit duration in seconds.
        weighted (bool): True if the training requires a weighted loss calculation.
        balanced (bool): True if the training requires a class-balanced sampling.
    """
    folds, unit_sec, weighted, folder, balanced = _defs[task]
    folder = folder or task
    return f'{METADATA_DIR}/{task}.csv', f'{WORK}/{_fs_table[cfg.sample_rate]}/{folder}', folds, unit_sec, weighted, balanced
