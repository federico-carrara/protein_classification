from pathlib import Path
from typing import Callable, Literal, Union

import numpy as np
import tifffile as tiff
from numpy.typing import NDArray
from tqdm import tqdm

PathLike = Union[Path, str]


def _compute_mean_std(
    data: NDArray, 
    strategy: Literal["channel-wise", "global"] = "channel-wise"
) -> tuple[NDArray, NDArray]:
    """
    Compute mean and standard deviation of an array.

    Parameters
    ----------
    data : NDArray
        Input data array. Expected input shape is (S, C, (Z), Y, X)
    norm_strategy : Literal["channel-wise", "global"]
        Normalization strategy. Default is "channel-wise".

    Returns
    -------
    tuple[NDArray, NDArray]
        Arrays of mean and standard deviation values per channel.
    """
    if strategy == "channel-wise":
        # Define the list of axes excluding the channel axis
        axes = tuple(np.delete(np.arange(data.ndim), 1))
        stats = (
            np.mean(data, axis=axes), # (C,)
            np.std(data, axis=axes) # (C,)
        )
    elif strategy == "global":
        axes = tuple(np.arange(data.ndim))
        stats = (
            np.asarray(np.mean(data, axis=axes))[None], # (1,)
            np.asarray(np.std(data, axis=axes))[None] # (1,)
        )
    else:
        raise ValueError(
            (
                f"Unknown normalization strategy: {strategy}."
                "Available ones are 'channel-wise' and 'global'."
            )
        )
    return stats


def update_iterative_stats(
    count: NDArray, mean: NDArray, m2: NDArray, new_values: NDArray
) -> tuple[NDArray, NDArray, NDArray]:
    """Update the mean and variance of an array iteratively.

    Parameters
    ----------
    count : NDArray
        Number of elements in the array. Shape: (C,).
    mean : NDArray
        Mean of the array. Shape: (C,).
    m2 : NDArray
        Variance of the array. Shape: (C,).
    new_values : NDArray
        New values to add to the mean and variance. Shape: (C, 1, 1, Z, Y, X).

    Returns
    -------
    tuple[NDArray, NDArray, NDArray]
        Updated count, mean, and variance.
    """
    num_channels = len(new_values)

    # --- update channel-wise counts ---
    count += np.ones_like(count) * np.prod(new_values.shape[1:])

    # --- update channel-wise mean ---
    # compute (new_values - old_mean) -> shape: (C, Z*Y*X)
    delta = new_values.reshape(num_channels, -1) - mean.reshape(num_channels, 1)
    mean += np.sum(delta / count.reshape(num_channels, 1), axis=1)

    # --- update channel-wise SoS ---
    # compute (new_values - new_mean) -> shape: (C, Z*Y*X)
    delta2 = new_values.reshape(num_channels, -1) - mean.reshape(num_channels, 1)
    m2 += np.sum(delta * delta2, axis=1)

    return count, mean, m2


def finalize_iterative_stats(
    count: NDArray,
    mean: NDArray,
    m2: NDArray,
    strategy: Literal["channel-wise", "global"]
) -> tuple[NDArray, NDArray]:
    """Finalize the mean and variance computation.

    Parameters
    ----------
    count : NDArray
        Number of elements in the array. Shape: (C,).
    mean : NDArray
        Mean of the array. Shape: (C,).
    m2 : NDArray
        Variance of the array. Shape: (C,).
    strategy : Literal["channel-wise", "global"]
        The type of normalization to be applied.

    Returns
    -------
    tuple[NDArray, NDArray]
        Final channel-wise mean and standard deviation.
    """
    std = np.sqrt(m2 / count) # (C,)
    if strategy == "channel-wise":
        if any(c < 2 for c in count):
            return np.full(mean.shape, np.nan), np.full(std.shape, np.nan)
        else:
            return mean, std
    elif strategy == "global":
        global_mean = np.mean(mean)
        global_std = np.sqrt(np.mean(std ** 2) + np.mean((mean - global_mean) ** 2))
        return global_mean[None], global_std[None]


class WelfordStatistics:
    """Compute Welford statistics iteratively.

    The Welford algorithm is used to compute the mean and variance of an array
    iteratively. Based on the implementation from:
    https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Welford's_online_algorithm
    """
    
    def __init__(self, strategy: Literal["channel-wise", "global"]) -> None:
        """Constructor for WelfordStatistics.
        
        Parameters
        ----------
        strategy : Literal["channel-wise", "global"]
            Strategy used to compute statistics for normalization, either
            "channel-wise" or "global".
        """
        self.strategy = strategy

    def update(self, array: NDArray, sample_idx: int) -> None:
        """Update the Welford statistics.

        Parameters
        ----------
        array : NDArray
            Input array of shape (S, C, (Z), Y, X).
        sample_idx : int
            Current sample number.
        """
        self.sample_idx = sample_idx
        sample_channels = np.array(np.split(array, array.shape[1], axis=1))

        # Initialize the statistics
        if self.sample_idx == 0:
            # Compute the mean and standard deviation
            stats = _compute_mean_std(array, self.strategy)
            self.mean = stats[0]
            # Initialize the count and m2 with zero-valued arrays of shape (C,)
            self.count, self.mean, self.m2 = update_iterative_stats(
                count=np.zeros(array.shape[1]),
                mean=self.mean,
                m2=np.zeros(array.shape[1]),
                new_values=sample_channels,
            )
        else:
            # Update the statistics
            self.count, self.mean, self.m2 = update_iterative_stats(
                count=self.count, mean=self.mean, m2=self.m2, new_values=sample_channels
            )

        self.sample_idx += 1

    def finalize(self) -> tuple[NDArray, NDArray]:
        """Finalize the Welford statistics.

        Returns
        -------
        tuple or numpy arrays
            Final mean and standard deviation.
        """
        return finalize_iterative_stats(self.count, self.mean, self.m2, self.strategy)


class RunningMinMaxStatistics:
    """Compute running min and max statistics.
    
    Min and max statistics are computed iteratively and updated for each input array.
    
    For robustness, the min and max values are taken as the 1% and 99% percentiles
    of the intensity values in the data.
    """
    
    def __init__(self, strategy: Literal["channel-wise", "global"]) -> None:
        """Constructor for RunningMinMaxStatistics.
        
        Parameters
        ----------
        strategy : Literal["channel-wise", "global"]
            Strategy used to compute statistics for normalization, either
            "channel-wise" or "global".
        """
        self.mins = None
        self.maxs = None
        self.strategy = strategy
    
    def update(self, array: NDArray) -> None:
        """Update the running min and max statistics.
        
        Parameters
        ----------
        array : NDArray
            Input array of shape (S, C, (Z), Y, X).
        """
        # TODO: make quantiles as a parameter!
        axes = tuple(np.delete(np.arange(array.ndim), 1))
        if self.mins is None:
            self.mins = np.quantile(array, 0.005, axis=axes) # (C,)
            self.maxs = np.quantile(array, 0.995, axis=axes) # (C,)
        else:
            self.mins = np.minimum(
                self.mins, np.quantile(array, 0.005, axis=axes)
            ) # (C,)
            self.maxs = np.maximum(
                self.maxs, np.quantile(array, 0.995, axis=axes)
            ) # (C,)
            
    def finalize(self) -> tuple[NDArray, NDArray]:
        """Finalize the running min and max statistics.
        
        Returns
        -------
        tuple or numpy arrays
            Final min and max values.
        """
        if self.strategy == "channel-wise":
            return self.mins, self.maxs
        elif self.strategy == "global":
            return np.min(self.mins)[None], np.max(self.maxs)[None]


def calculate_dataset_stats(
    filepaths: PathLike, imreader: Callable[[PathLike], NDArray] = tiff.imread,
) -> dict[str, float]:
    """Calculate running dataset statistics (mean, std, min, max).
    
    Parameters
    ----------
    filepaths : PathLike
        Paths to the image files in the dataset.
    imreader : Callable[[PathLike], NDArray], optional
        Function to read the image files. Default is `tiff.imread`.

    Returns
    -------
    dict[str, float]
        Dictionary containing the dataset statistics.
    """
    num_samples = 0
    welford_stats = WelfordStatistics("global")
    minmax_stats = RunningMinMaxStatistics("global")
    for fpath in tqdm(filepaths):
        img = imreader(fpath)[np.newaxis, np.newaxis, ...]
        welford_stats.update(img, num_samples)
        minmax_stats.update(img)
        num_samples += 1

    if num_samples == 0:
        raise ValueError("No samples found in the dataset.")

    # Get statistics
    image_means, image_stds = welford_stats.finalize()
    image_mins, image_maxs = minmax_stats.finalize()
    return {
        "mean": image_means.item(),
        "std": image_stds.item(),
        "min": image_mins.item(),
        "max": image_maxs.item(),
    }