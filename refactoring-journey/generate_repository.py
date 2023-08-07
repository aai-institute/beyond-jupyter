import os
import shutil
from git import Repo


def create_or_update_repo_with_tags():
    """
    Create or update a Git repository with tags representing different steps.

    This function initializes a Git repository in the destination folder, then iterates over all subdirectories
    in the source folder that start with 'step', creating a commit and tag for each one.

    If the destination folder already exists, it will be deleted and a new repository will be created.
    """
    source_folder = os.path.dirname(os.path.abspath(__file__))
    destination_folder = os.path.join(source_folder, 'refactoring_repo')

    if os.path.exists(destination_folder):
        shutil.rmtree(destination_folder)

    repo = Repo.init(destination_folder)

    step_folders = [f for f in os.listdir(source_folder) if f.startswith('step') and os.path.isdir(os.path.join(source_folder, f))]
    step_folders.sort()

    for step_folder_name in step_folders:
        source_step_folder = os.path.join(source_folder, step_folder_name)
        destination_step_folder = os.path.join(destination_folder, "step")

        for filename in os.listdir(destination_folder):
            dir_item = os.path.join(destination_folder, filename)
            if filename.startswith('step') and os.path.isdir(dir_item):
                shutil.rmtree(dir_item)

        shutil.copytree(source_step_folder, destination_step_folder)

        repo.index.add('*')
        repo.index.commit(f'Step: {step_folder_name}')
        repo.create_tag(f'{step_folder_name}')

    print(f'Repository created and tagged in {destination_folder}')


if __name__ == '__main__':
    create_or_update_repo_with_tags()
