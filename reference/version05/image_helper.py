import os


def get_image_paths(directory, num_samples=None):
    """
    Returns a list of paths to images found at the specified directory up to num_samples.

    :param directory: Directory where the images are located.
    :param num_samples: If specified, this function will limit the result to first num_samples elements.
    :return: List of paths to images found at the specified directory.
    """
    image_paths = [os.path.join(directory, name) for name in os.listdir(directory) if name.endswith(".jpg")]
    if num_samples is not None:
        image_paths = image_paths[:num_samples]
    return image_paths

