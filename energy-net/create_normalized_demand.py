import numpy as np


def aggregate_pcs_actions(file_paths, aggregation="sum", weights=None, clip=True, clip_min=-1.0, clip_max=1.0, align="min", scales=None, biases=None, save_path=None):
    """
    Aggregate multiple PCS action sequences from .npy files into a single sequence.

    Parameters
    ----------
    file_paths : list[str]
        Paths to .npy files, each containing a 1D array of PCS actions.
    aggregation : {"sum", "mean", "weighted"}
        Aggregation method across sequences.
    weights : list[float] | None
        Weights for the "weighted" aggregation method. Must match number of files.
    clip : bool
        Whether to clip the aggregated actions to [clip_min, clip_max].
    clip_min : float
        Minimum value for clipping.
    clip_max : float
        Maximum value for clipping.
    align : {"min", "max"}
        How to align different-length sequences:
        - "min": trim all to the length of the shortest sequence
        - "max": pad shorter sequences with zeros to match the longest sequence

    Scaling / Denormalization
    -------------------------
    If your inputs are normalized (e.g., in [-1, 1]) and you want physical units,
    provide per-file `scales` (and optional `biases`). Each sequence i will be
    transformed as: seq_i = seq_i * scales[i] + (biases[i] if provided else 0).

    Returns
    -------
    np.ndarray
        1D array of aggregated actions.
    """
    if not file_paths:
        raise ValueError("file_paths must be a non-empty list of .npy paths")

    # Load and normalize arrays to 1D
    sequences = []
    for path in file_paths:
        arr = np.load(path)
        if arr.ndim > 1:
            arr = np.squeeze(arr)
        if arr.ndim != 1:
            raise ValueError(f"Actions in '{path}' must be 1D after squeeze; got shape {arr.shape}")
        sequences.append(arr.astype(np.float32))

    lengths = [len(a) for a in sequences]
    min_len = int(min(lengths))
    max_len = int(max(lengths))

    if align == "min":
        sequences = [a[:min_len] for a in sequences]
        target_len = min_len
    elif align == "max":
        target_len = max_len
        padded = []
        for a in sequences:
            if len(a) < target_len:
                pad_width = target_len - len(a)
                a = np.pad(a, (0, pad_width), mode="constant", constant_values=0.0)
            else:
                a = a[:target_len]
            padded.append(a)
        sequences = padded
    else:
        raise ValueError("align must be one of {'min', 'max'}")

    # Optional per-file affine transform for denormalization: a*x + b
    if scales is not None:
        if np.isscalar(scales):
            scales = [float(scales)] * len(sequences)
        if len(scales) != len(sequences):
            raise ValueError("scales length must match number of input files (or be a scalar)")
        sequences = [seq * float(s) for seq, s in zip(sequences, scales)]
    if biases is not None:
        if np.isscalar(biases):
            biases = [float(biases)] * len(sequences)
        if len(biases) != len(sequences):
            raise ValueError("biases length must match number of input files (or be a scalar)")
        sequences = [seq + float(b) for seq, b in zip(sequences, biases)]

    stack = np.stack(sequences, axis=0)  # shape: (num_sequences, target_len)

    if aggregation == "sum":
        agg = np.sum(stack, axis=0)
    elif aggregation == "mean":
        agg = np.mean(stack, axis=0)
    elif aggregation == "weighted":
        if weights is None:
            raise ValueError("weights must be provided for 'weighted' aggregation")
        if len(weights) != stack.shape[0]:
            raise ValueError("weights length must match number of input files")
        w = np.asarray(weights, dtype=np.float32).reshape(-1, 1)
        agg = np.sum(stack * w, axis=0)
    else:
        raise ValueError("aggregation must be one of {'sum', 'mean', 'weighted'}")

    if clip:
        agg = np.clip(agg, clip_min, clip_max)

    if save_path is not None:
        if save_path.endswith(".npy"):
            np.save(save_path, agg)
        else:
            np.savetxt(save_path, agg, delimiter=",")

    return agg.astype(np.float32)







actions = aggregate_pcs_actions(
    file_paths=[
        "vladimir/b1_actions.npy",
        "vladimir/b2_actions.npy",
        "vladimir/b3_actions.npy",
        "vladimir/b4_actions.npy",
        "vladimir/b5_actions.npy",
        "vladimir/b6_actions.npy"
    ],
    aggregation="mean",         # was "sum"
    weights=None,
    clip=True,                  # was False
    clip_min=-5.0,              # set to env PCS bounds
    clip_max=5.0,
    align="min",
    save_path="vladimir/aggregated_actions_clipped.npy"  # auto-saves

)