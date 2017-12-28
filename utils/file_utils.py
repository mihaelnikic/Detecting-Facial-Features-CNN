import dill


def extend(file_name: str, extension: str):
    if file_name.endswith('.' + extension):
        return file_name
    else:
        return file_name + '.' + extension


def save_object(obj: object, save_file: str):
    with open(save_file, 'wb') as handle:
        dill.dump(obj, handle, protocol=dill.HIGHEST_PROTOCOL)


def load_object(load_file: str):
    with open(load_file, 'rb') as handle:
        obj = dill.load(handle)
    return obj
