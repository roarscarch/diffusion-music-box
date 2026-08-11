import numpy as np


class TilePipeline:
    """Process spectrogram tiles for the diffusion music box.

    This pipeline applies a sequence of operations to a spectrogram tile,
    such as normalization, spectral gating, contrast enhancement, and
    denoising. It is designed to be extensible so that new processing steps
    can be added easily. The pipeline is used to prepare tiles before
    inverse transform to audio.

    Parameters
    ----------
    steps : list of callables, optional
        A list of processing functions. Each function should take a 2D
        numpy array (freq x time) and return a processed 2D numpy array.
        If None, a default set of steps is used.
    normalize : bool, optional
        Whether to normalize the tile to [-1, 1] after processing.
        Default is True.
    """

    def __init__(self, steps=None, normalize=True):
        self.steps = steps if steps is not None else self._default_steps()
        self.normalize = normalize

    def _default_steps(self):
        """Return the default processing steps.

        The default pipeline applies a soft clipping to reduce harsh peaks,
        then a simple spectral gate to remove low-level noise, and finally
        a mild contrast enhancement to bring out texture.
        """
        return [
            self._soft_clip,
            self._spectral_gate,
            self._contrast_enhance,
        ]

    @staticmethod
    def _soft_clip(tile, threshold=0.9):
        """Apply soft clipping to limit extreme values.

        Values above threshold are compressed using a tanh-like curve.
        """
        return np.tanh(tile / threshold) * threshold

    @staticmethod
    def _spectral_gate(tile, floor_db=-60):
        """Gate low-level noise by zeroing values below a floor.

        The floor is specified in dB relative to the maximum amplitude.
        """
        max_val = np.max(np.abs(tile))
        if max_val == 0:
            return tile
        threshold = max_val * (10 ** (floor_db / 20))
        return np.where(np.abs(tile) < threshold, 0, tile)

    @staticmethod
    def _contrast_enhance(tile, factor=1.2):
        """Enhance contrast by scaling the deviation from the mean.

        This makes quiet parts quieter and loud parts louder, increasing
        the dynamic range.
        """
        mean = np.mean(tile)
        return (tile - mean) * factor + mean

    def process(self, tile):
        """Run the tile through the pipeline.

        Parameters
        ----------
        tile : np.ndarray
            2D float array of shape (freq_bins, time_steps).

        Returns
        -------
        np.ndarray
            Processed tile, normalized to [-1, 1] if ``normalize`` is True.
        """
        result = np.asarray(tile, dtype=np.float32)
        if result.ndim != 2:
            raise ValueError("Tile must be 2D (freq x time)")

        for step in self.steps:
            result = step(result)

        if self.normalize:
            max_val = np.max(np.abs(result))
            if max_val > 0:
                result = result / max_val

        return result

    def add_step(self, func, index=None):
        """Add a new processing step to the pipeline.

        Parameters
        ----------
        func : callable
            A function that takes a 2D numpy array and returns a processed
            2D numpy array.
        index : int, optional
            Position at which to insert the step. If None, append to the end.
        """
        if index is None:
            self.steps.append(func)
        else:
            self.steps.insert(index, func)

    def remove_step(self, func):
        """Remove a processing step from the pipeline.

        Parameters
        ----------
        func : callable
            The function to remove. If not found, raises ValueError.
        """
        try:
            self.steps.remove(func)
        except ValueError:
            raise ValueError("Step not found in pipeline") from None

    def clear_steps(self):
        """Remove all processing steps."""
        self.steps = []

    def __call__(self, tile):
        """Convenience method to process a tile."""
        return self.process(tile)
