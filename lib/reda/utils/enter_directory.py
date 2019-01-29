import os


class EnterDirectory(object):
    """This is a context manager that enters a given directory and returns to
    the initial current directory after finishing
    """
    def __init__(self, directory):
        self.pwd = os.getcwd()
        self.directory = os.path.abspath(str(directory))

    def __enter__(self):
        os.chdir(self.directory)

    def __exit__(self, dtype, value, traceback):
        os.chdir(self.pwd)


class CreateEnterDirectory(EnterDirectory):
    """This is a context manager that enters a given directory and returns to
    the initial current directory after finishing. If the target directory does
    not exist, create it.
    """
    def __enter__(self):
        os.makedirs(self.directory, exist_ok=True)
        os.chdir(self.directory)
