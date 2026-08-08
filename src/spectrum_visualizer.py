import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import threading


class SpectrumVisualizer:
    """Real-time visualization of spectrogram tiles during generation.

    Displays the current spectrogram tile as a heatmap in a separate thread.
    This is useful for monitoring the diffusion process and debugging the
    generation pipeline.

    Parameters
    ----------
    refresh_interval_ms : int
        Milliseconds between frame updates.
    colormap : str
        Matplotlib colormap name for the heatmap.
    """

    def __init__(self, refresh_interval_ms=100, colormap='magma'):
        self.refresh_interval_ms = refresh_interval_ms
        self.colormap = colormap
        self._fig = None
        self._ax = None
        self._im = None
        self._thread = None
        self._lock = threading.Lock()
        self._latest_tile = None
        self._running = False

    def start(self):
        """Start the visualization in a background thread."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        """Stop the visualization thread and close the figure."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=1.0)
        if self._fig is not None:
            plt.close(self._fig)
            self._fig = None

    def update_tile(self, tile):
        """Update the displayed spectrogram tile.

        Parameters
        ----------
        tile : np.ndarray
            2D spectrogram tile (frequencies x time) to display.
        """
        with self._lock:
            self._latest_tile = np.asarray(tile, dtype=np.float32)

    def _run(self):
        """Run the matplotlib animation loop."""
        plt.ion()
        self._fig, self._ax = plt.subplots(figsize=(8, 6))
        self._im = self._ax.imshow(
            np.zeros((1, 1)),
            aspect='auto',
            cmap=self.colormap,
            origin='lower'
        )
        self._ax.set_xlabel('Time frames')
        self._ax.set_ylabel('Frequency bins')
        self._ax.set_title('Spectrogram Tile')
        self._fig.colorbar(self._im, ax=self._ax)

        def animate(frame):
            with self._lock:
                tile = self._latest_tile
            if tile is not None:
                self._im.set_data(tile)
                self._im.set_clim(vmin=np.min(tile), vmax=np.max(tile))
            return [self._im]

        ani = FuncAnimation(
            self._fig,
            animate,
            interval=self.refresh_interval_ms,
            blit=False,
            cache_frame_data=False
        )
        plt.show()
        self._running = False
