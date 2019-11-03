# from keras.preprocessing.image import img_to_array, load_img
from pathlib import Path
import itertools
import h5py


# def load_img_as_array(filepath):
#     """学習時に合わせて画像サイズを224*224とする
#     """
#     arrayimg = img_to_array(load_img(filepath, target_size=(224, 224)))
#     return arrayimg


def list_pictures(directory, exts=None):
    """指定したディレクトリにある拡張子extのファイルをリストとして返す

    Arguments:
        directory {str} -- 探索するディレクトリ

    Keyword Arguments:
        exts {str list} -- 探索するファイルの拡張子

    Returns:
        img_list {str list} -- 見つけたファイルパスのリスト
    """
    if exts is None:
        exts = ["jpg", "JPG", "jpeg", "JPEG"]

    dirpath = Path(directory)
    img_gens = [dirpath.glob('*.' + ext) for ext in exts]
    img_list = [path for path in itertools.chain(*img_gens)]
    img_list = sorted(img_list)
    return img_list


def list_files_recursively(directory, exts):
    """指定したディレクトリにある拡張子extのファイルをリストとして返す

    Arguments:
        directory {str} -- 探索するディレクトリ
        ext {str} -- 探索するファイルの拡張子 (例: {["jpg","JPG"]})

    Returns:
        img_list {str list} -- 見つけたファイルパスのリスト
    """

    # TODO: cacheを無視できるようにする（list_subdir_filesに変更）
    dirpath = Path(directory)
    file_gens = [dirpath.glob('**/*.' + ext) for ext in exts]
    file_list = [path for path in itertools.chain(*file_gens)]
    file_list = sorted(file_list)
    return file_list


class CacheStorage(object):
    def __init__(self):
        self.cache_path = Path("./cache/cache.h5")

    def has(self, path):
        path = str(path)
        exist = False
        if self.cache_path.exists():
            with h5py.File(self.cache_path, "r") as f:
                exist = path in f
        else:
            with h5py.File(self.cache_path, "w") as f:
                pass
        return exist

    def set_cache(self, path, value, override=False):
        path = str(path)
        if self.cache_path.exists():
            with h5py.File(self.cache_path, "r+") as f:
                f.create_dataset(path, data=value)
        else:
            with h5py.File(self.cache_path, "w") as f:
                f.create_dataset(path, data=value)

    def get_cache(self, path):
        path = str(path)
        cache = None
        if self.cache_path.exists() and self.has(path):
            with h5py.File(self.cache_path, "r") as f:
                cache = f[path][()]
        return cache
