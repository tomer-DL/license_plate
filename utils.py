import configparser
import os

def read_section(file_name, section):
    config = configparser.ConfigParser()
    config.read(file_name)
    dictionary = {}
    for option in config.options(section):
        dictionary[option] = config.get(section, option)

    return dictionary


def file_generator(root, ext="", recursive=False):
    for x in os.listdir(root):
        if os.path.isfile(root + "\\" + x) and x.endswith(ext):
            yield root + "\\" + x
        elif recursive and os.path.isdir(root + "\\" + x):
            yield from file_generator(root + "\\" + x, ext, True)
