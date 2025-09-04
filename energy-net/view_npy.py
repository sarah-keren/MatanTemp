import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path



FILE_PATH = Path("files_for_demo/iso_actions_intervals.npy")  # update if needed

def extract_two_prices(obj):
    """
    Try to extract two price series from common .npy structures:
      - 2D numeric array: take first two columns
      - 1D array with interleaved values: split even/odd indices
      - Object array of dicts: take two keys containing 'price'
      - Structured array with named fields: take two fields (prefer those containing 'price')
      - Dict: take two numeric arrays (prefer keys containing 'price')
    Returns: (p1, p2, labels)
    """
    def try_numeric_2d(arr_like):
        try:
            arr = np.asarray(arr_like, dtype=float)
            if arr.ndim == 2 and arr.shape[1] >= 2:
                return arr[:, 0], arr[:, 1], ("col0", "col1")
        except Exception:
            pass
        return None

    # NumPy ndarray cases
    if isinstance(obj, np.ndarray):
        # Structured array with named fields
        if obj.dtype.names is not None:
            fields = list(obj.dtype.names)
            price_fields = [f for f in fields if "price" in f.lower()]
            if len(price_fields) >= 2:
                return np.asarray(obj[price_fields[0]], dtype=float), \
                       np.asarray(obj[price_fields[1]], dtype=float), \
                       (price_fields[0], price_fields[1])
            if len(fields) >= 2:
                return np.asarray(obj[fields[0]], dtype=float), \
                       np.asarray(obj[fields[1]], dtype=float), \
                       (fields[0], fields[1])

        # Object array (e.g., list of dicts)
        if obj.dtype == object:
            # Object array of dicts
            if len(obj) and isinstance(obj[0], dict):
                keys = set()
                for d in obj[: min(10, len(obj))]:
                    keys.update(d.keys())
                price_keys = [k for k in keys if "price" in str(k).lower()]
                if len(price_keys) >= 2:
                    p1 = np.array([float(d.get(price_keys[0], np.nan)) for d in obj], dtype=float)
                    p2 = np.array([float(d.get(price_keys[1], np.nan)) for d in obj], dtype=float)
                    return p1, p2, (str(price_keys[0]), str(price_keys[1]))
                # fallback: first two numeric keys
                numeric_keys = []
                for k in keys:
                    try:
                        vals = [float(d.get(k, np.nan)) for d in obj]
                        numeric_keys.append((k, vals))
                    except Exception:
                        pass
                if len(numeric_keys) >= 2:
                    return np.array(numeric_keys[0][1], dtype=float), \
                           np.array(numeric_keys[1][1], dtype=float), \
                           (str(numeric_keys[0][0]), str(numeric_keys[1][0]))
            # Object array of lists/tuples -> try casting to 2D
            try:
                arr2d = np.array(obj.tolist(), dtype=float)
                pair = try_numeric_2d(arr2d)
                if pair:
                    return pair
            except Exception:
                pass

        # Plain numeric 2D array
        pair = try_numeric_2d(obj)
        if pair:
            return pair

        # 1D numeric array: treat even/odd as two series (heuristic)
        try:
            arr1d = np.asarray(obj, dtype=float)
            if arr1d.ndim == 1 and arr1d.size >= 2:
                return arr1d[0::2], arr1d[1::2], ("series_even_idx", "series_odd_idx")
        except Exception:
            pass

    # Dict-like
    if isinstance(obj, dict):
        price_keys = [k for k in obj.keys() if "price" in str(k).lower()]
        def is_num_vec(v):
            try:
                a = np.asarray(v, dtype=float)
                return a.ndim == 1 and a.size > 0
            except Exception:
                return False
        if len(price_keys) >= 2 and is_num_vec(obj[price_keys[0]]) and is_num_vec(obj[price_keys[1]]):
            return np.asarray(obj[price_keys[0]], dtype=float), \
                   np.asarray(obj[price_keys[1]], dtype=float), \
                   (str(price_keys[0]), str(price_keys[1]))
        numeric_items = [(k, np.asarray(v, dtype=float)) for k, v in obj.items() if is_num_vec(v)]
        if len(numeric_items) >= 2:
            return numeric_items[0][1], numeric_items[1][1], (str(numeric_items[0][0]), str(numeric_items[1][0]))

    raise ValueError("Could not infer two price series from iso_actions_intervals.npy")

def main():
    data = np.load(FILE_PATH, allow_pickle=True)
    p1, p2, labels = extract_two_prices(data)

    # First 48 intervals
    n = 100
    p1_48 = p1[:n]
    p2_48 = p2[:n]

    # Plot
    x = np.arange(len(p1_48))
    plt.figure()
    plt.plot(x, p1_48, label=labels[0])
    plt.plot(x, p2_48, label=labels[1])
    plt.title("Two Prices")
    plt.xlabel("Interval")
    plt.ylabel("Price")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Save as PNG too (optional)
    out = "price_plot.png"
    plt.figure()
    plt.plot(x, p1_48, label=labels[0])
    plt.plot(x, p2_48, label=labels[1])
    plt.title("Two Prices")
    plt.xlabel("Interval")
    plt.ylabel("Price")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    print(f"Saved: {out}")

if __name__ == "__main__":
    arr = np.load("files_for_demo/iso_actions_intervals.npy")
    print(pd.DataFrame(arr))
    main()

