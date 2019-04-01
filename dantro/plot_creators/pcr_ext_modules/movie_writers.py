"""This module implements Movie Writer utilities"""

import os

import matplotlib as mpl
import matplotlib.animation
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------

@mpl.animation.writers.register('frames')
class FileWriter(mpl.animation.AbstractMovieWriter):
    """A matplotlib file writer

    It adheres to the corresponding matplotlib animation interface.
    """

    def __init__(self, *, name_padding: int=7,
                 fstr: str="{dir:}/{num:0{pad:}d}.{ext:}"):
        """
        Initialize a FileWriter, which adheres to the matplotlib.animation
        interface and can be used to write individual files.

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
        self.file_format = None

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

    def saving(self, fig, base_outfile: str, dpi: int=None, **setup_kwargs):
        """Create an instance of the context manager
        
        Args:
            fig (matplotlib.Figure): The figure object to save
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
        self.file_format = ext[1:]  # includes leading dot

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
        outfile = self.fstr.format(dir=self.out_dir,
                                   num=self.cntr,
                                   pad=self.name_padding,
                                   ext=self.file_format)

        # Save the frame using the context manager, then increment the cntr
        self.fig.savefig(outfile, dpi=self.dpi, file_format=self.file_format,
                         **savefig_kwargs)
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
        plt.close(self.fig)
