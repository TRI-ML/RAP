import os

from pygit2 import Repository, discover_repository

current_working_directory = os.getcwd()
repository_path = discover_repository(current_working_directory)
repo = Repository(repository_path)