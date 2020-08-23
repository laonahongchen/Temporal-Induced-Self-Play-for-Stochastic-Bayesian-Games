import os


def join_path(path_a, path_b):
    return os.path.join(path_a, path_b)


def check_path(path):
    dirname = os.path.dirname(path)
    if any(dirname):
        os.makedirs(dirname, exist_ok=True)


def join_path_and_check(path_a, path_b):
    path = join_path(path_a, path_b)
    check_path(path)
    return path
