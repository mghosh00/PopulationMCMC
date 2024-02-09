#
# Class to plot aspects of the population MCMC (helper class)
#
import pandas as pd
from plotnine import ggplot, aes, geom_line, geom_point, geom_path, ggtitle


class Plotter:
    """A static class for plotting
    """

    @staticmethod
    def plot_traces(df: pd.DataFrame, title: str):
        """Creates a plot of traces, coloured by `chain._id`.

        Parameters
        ----------
        df : pd.DataFrame
            Data containing the iteration timestep and parameter values
        title : str
            Title of the plots file
        """
        column_names = df.columns
        for param in column_names[1:-1]:
            plot = (ggplot(df, aes(x='t', y=param)) +
                    geom_line(aes(color='factor(id)')) +
                    ggtitle(f"Traces for {param} ({title})")
                    )
            plot.save(f"plots/{title}_traces_{param}.png")
