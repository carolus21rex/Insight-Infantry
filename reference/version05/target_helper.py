import os


def get_target_paths(directory, num_samples=None):
    """
    Returns a list of paths to targets found at the specified directory up to num_samples.

    :param directory: Directory where the target files are located.
    :param num_samples: If specified, this function will limit the result to first num_samples elements.
    :return: List of paths to targets files found at the specified directory.
    """
    target_paths = [os.path.join(directory, name) for name in os.listdir(directory) if name.endswith(".txt")]
    if num_samples is not None:
        target_paths = target_paths[:num_samples]
    return target_paths
