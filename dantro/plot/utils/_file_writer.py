"""This module implements custom matplotlib movie writers; basically, these
are specializations of :py:class:`matplotlib.animation.AbstractMovieWriter`."""

import os

import matplotlib as mpl
import matplotlib.animation
import matplotlib.figure

# -----------------------------------------------------------------------------


@mpl.animation.writers.register("frames")
class FileWriter(mpl.animation.AbstractMovieWriter):
    """A specialization of :py:class:`matplotlib.animation.AbstractMovieWriter`
    that writes each frame to a file.

    It is registered as the ``frames`` writer.
    """

    def __init__(
        self,
        *,
        name_padding: int = 7,
        fstr: str = "{dir:}/{num:0{pad:}d}.{ext:}",
    ):
        """
        Initialize the FileWriter, which adheres to the
        :py:mod:`matplotlib.animation` interface and can be used to write
        each frame of an animation to individual files.

        Args:
            name_padding (int, optional): How wide the numbering should be
            fstr (str, optional): The format string to generate the name
        """
        self.cntr = 0
        self.name_padding = name_padding
        self.fstr = fstr

        # Other attributes that are to be determined later
        self.fig = None
        self.out_dir = None
        self.dpi = None
        self.format = None

    def setup(self):
        """Called when entering the saving context"""
        pass

    def finish(self):
        """Called when finished"""
        pass

    @classmethod
    def isAvailable(cls) -> bool:
        """Always available."""
        return True

    def saving(
        self,
        fig: mpl.figure.Figure,
        base_outfile: str,
        dpi: int = None,
        **setup_kwargs,
    ):
        """Create an instance of the context manager

        Args:
            fig (matplotlib.figure.Figure): The figure object to save
            base_outfile (str): The path this movie writer would store a movie
                file at; the file name will be interpreted as the name of the
                directory that the frames are saved to; the file extension
                is retained.
            dpi (int, optional): The desired densiy
            **setup_kwargs: Passed to setup method

        Returns:
            FileWriter: this object, which also is a context manager.
        """
        # Parse the given base file path to get a directory and extension
        out_dir, ext = os.path.splitext(base_outfile)
        self.format = ext[1:]  # includes leading dot

        # Store all required objects
        self.fig = fig
        self.dpi = dpi if dpi is not None else self.fig.dpi
        self.out_dir = out_dir

        # Now, call the setup function
        self.setup(**setup_kwargs)

        # As this writer is itself a context manager, return self
        return self

    def grab_frame(self, **savefig_kwargs):
        """Stores a single frame"""
        # Build the output path from the info of the context manager
        outfile = self.fstr.format(
            dir=self.out_dir,
            num=self.cntr,
            pad=self.name_padding,
            ext=self.format,
        )

        # Save the frame using the context manager, then increment the cntr
        self.fig.savefig(
            outfile,
            dpi=self.dpi,
            format=self.format,
            **savefig_kwargs,
        )
        self.cntr += 1

    # .. Context manager magic methods ........................................

    def __enter__(self):
        """Called when entering context.

        Makes sure that the output directory exists.
        """
        os.makedirs(self.out_dir, exist_ok=True)

    def __exit__(self, *args):
        """Called when exiting context.

        Closes the figure.
        """
        import matplotlib.pyplot as plt

        plt.close(self.fig)
