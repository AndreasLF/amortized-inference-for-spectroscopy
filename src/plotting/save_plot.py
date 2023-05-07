import os

def save_plot(fig, path, extensions=["eps", "png"]):
    """Save figure to path with given extension
    
    Args:
        fig (matplotlib.figure.Figure): figure to save
        path (str): path to save figure to
        extension (list, optional): list of extensions to save figure to. Defaults to ["eps", "png"].

    Returns:
        None
    """
    split = os.path.split(path)
    # folders, everything in split except last element
    folders = split[:-1]
    filename = split[-1]
    # only get name without extension 
    filename = os.path.splitext(filename)[0]

    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    report_dir = os.path.join(parent_dir, 'report')
    # create methods folder in figures_path
    if not os.path.exists(os.path.join(report_dir, *folders)):
        os.makedirs(os.path.join(report_dir, *folders))

    # save figure to methods folder
    for ext in extensions:
        # remove file if it already exists
        if os.path.exists(os.path.join(report_dir, *folders, filename + "." + ext)):
            os.remove(os.path.join(report_dir, *folders, filename + "." + ext))
        # overwrite existing files
        fig.savefig(os.path.join(report_dir, *folders, filename + "." + ext), bbox_inches='tight', format=ext, dpi=300)
