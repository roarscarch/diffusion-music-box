import numpy as np
import matplotlib.pyplot as plt


class SpectrogramVisualizer:
    """Visualize spectrogram tiles for debugging and monitoring.

    This class provides a simple interface to display 2D spectrogram tiles
    as heatmaps. It is intended for offline analysis and debugging of the
    generated spectrograms, not for real-time display during playback.
    """

    def __init__(self, cmap='viridis', figsize=(10, 6)):
        self.cmap = cmap
        self.figsize = figsize

    def show(self, spectrogram, title='Spectrogram', save_path=None):
        """Display a 2D spectrogram as a heatmap.

        Parameters
        ----------
        spectrogram : np.ndarray
            2D array of shape (freq_bins, time_frames).
        title : str, optional
            Title of the plot.
        save_path : str, optional
            If provided, save the figure to this path instead of showing.
        """
        if spectrogram.ndim != 2:
            raise ValueError("Spectrogram must be 2D")

        fig, ax = plt.subplots(figsize=self.figsize)
        im = ax.imshow(spectrogram, aspect='auto', cmap=self.cmap, origin='lower')
        ax.set_xlabel('Time frames')
        ax.set_ylabel('Frequency bins')
        ax.set_title(title)
        plt.colorbar(im, ax=ax)

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
        else:
            plt.show()

    def show_magnitude(self, spectrogram, title='Spectrogram (Magnitude)', save_path=None):
        """Display the magnitude (absolute value) of a spectrogram.

        Parameters
        ----------
        spectrogram : np.ndarray
            2D array of complex or real values.
        title : str, optional
            Title of the plot.
        save_path : str, optional
            If provided, save the figure to this path instead of showing.
        """
        magnitude = np.abs(spectrogram)
        self.show(magnitude, title=title, save_path=save_path)

    def save_animation(self, spectrogram_sequence, output_path, fps=10, title='Spectrogram Animation'):
        """Save an animation of a sequence of spectrograms as a GIF.

        Parameters
        ----------
        spectrogram_sequence : list of np.ndarray
            List of 2D spectrogram tiles.
        output_path : str
            Path to save the GIF.
        fps : int, optional
            Frames per second.
        title : str, optional
            Title for each frame.
        """
        try:
            import matplotlib.animation as animation
        except ImportError:
            raise ImportError("matplotlib is required for animation")

        if not spectrogram_sequence:
            raise ValueError("No spectrograms provided")

        fig, ax = plt.subplots(figsize=self.figsize)
        im = ax.imshow(np.abs(spectrogram_sequence[0]), aspect='auto', cmap=self.cmap, origin='lower')
        ax.set_title(title)
        plt.colorbar(im, ax=ax)

        def update(frame_idx):
            im.set_array(np.abs(spectrogram_sequence[frame_idx]))
            return [im]

        ani = animation.FuncAnimation(fig, update, frames=len(spectrogram_sequence), interval=1000/fps)
        ani.save(output_path, writer='pillow')
        plt.close(fig)
