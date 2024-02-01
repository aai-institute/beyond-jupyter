from pathlib import Path


def csv_data_path() -> Path:
    """
    Returns the location of the Spotify CSV data, allowing for script executions in subfolders without worrying about the
    relative location of the data

    :return: the path to the CSV file
    """
    cwd = Path("..")
    for folder in (cwd, cwd / "..", cwd / ".." / ".."):
        data_folder = folder / "data"
        if data_folder.exists() and data_folder.is_dir():
            return data_folder / "spotify_data.csv"
