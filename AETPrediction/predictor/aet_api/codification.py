import hashlib
import pandas as pd
import numpy as np

def categorialcoding(val):
    """
    Encode categorical values using MD5 hash.
    Handles both individual values and pandas Series.
    """
    if isinstance(val, pd.Series):
        # Handle Series with vectorized operations
        result = pd.Series(index=val.index, dtype='int64')
        # Fill NaN values with -1
        result[val.isna()] = -1
        # Apply hash encoding to non-NaN values
        non_na_mask = ~val.isna()
        if non_na_mask.any():
            str_values = val[non_na_mask].astype(str)
            hash_values = str_values.apply(lambda x: int(hashlib.md5(x.encode()).hexdigest(), 16) % (10**8))
            result[non_na_mask] = hash_values
        return result
    else:
        # Handle individual values
        if pd.isna(val):  # handle NaNs
            return -1
        return int(hashlib.md5(str(val).encode()).hexdigest(), 16) % (10**8)