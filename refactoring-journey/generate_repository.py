import os
import shutil
from pathlib import Path

from git import Repo


IGNORED_NAMES = {"README.md", "results", "mlruns", "res"}


def create_or_update_repo_with_tags():
    """
    Create or update a Git repository with tags representing different steps.

    This function initializes a Git repository in the destination folder, then iterates over all subdirectories
    in the source folder that start with 'step', creating a commit and tag for each one.

    If the destination folder already exists, it will be deleted and a new repository will be created.
    """
    root_folder = os.path.dirname(os.path.abspath(__file__))
    repo_folder = os.path.join(root_folder, 'refactoring_repo')

    if os.path.exists(repo_folder):
        shutil.rmtree(repo_folder)

    repo = Repo.init(repo_folder)
    os.chdir(repo_folder)

    # add .gitignore
    shutil.copy(Path(root_folder) / ".." / ".gitignore", repo_folder)
    repo.index.add(".gitignore")
    repo.index.commit("Add .gitignore")

    step_folders = [f for f in os.listdir(root_folder) if f.startswith('step') and os.path.isdir(os.path.join(root_folder, f))]
    step_folders.sort()

    for step_folder_name in step_folders:
        source_step_folder = Path(root_folder) / step_folder_name
        destination_step_folder = Path(repo_folder) / "step"

        # remove old step folder (if any)
        if destination_step_folder.is_dir():
            shutil.rmtree(destination_step_folder)
        destination_step_folder.mkdir(parents=True, exist_ok=False)

        # copy new step folder items
        for item in source_step_folder.iterdir():
            if item.name in IGNORED_NAMES:
                continue
            dest_item = destination_step_folder / item.name
            print(f"Copying {item} to {dest_item}")
            if item.is_dir():
                shutil.copytree(item, dest_item)
            else:
                shutil.copy(item, dest_item)

        #repo.index.add('*', force=False)  # This does not take .gitignore into consideration and will commit .pyc files and so on
        os.system("git add .")
        repo.index.commit(f'Step: {step_folder_name}')
        repo.create_tag(f'{step_folder_name}')


    print(f'Repository created and tagged in {repo_folder}')


if __name__ == '__main__':
    create_or_update_repo_with_tags()
