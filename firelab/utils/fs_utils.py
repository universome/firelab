import os
import shutil


def clean_dir(dirpath, create=False):
    if not os.path.exists(dirpath):
        if create: os.mkdir(dirpath)
        return

    for filename in os.listdir(dirpath):
        filepath = os.path.join(dirpath, filename)

        try:
            shutil.rmtree(filepath)
        except OSError:
            os.remove(filepath)


def clean_file(filepath, create=False):
    if not os.path.exists(filepath):
        touch_file(filepath)
    else:
        open(filepath, 'w').close()


def touch_file(file_path):
    open(file_path, 'a').close()
