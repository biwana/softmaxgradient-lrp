import numpy as np
from pathlib import Path
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.vgg19 import VGG19, preprocess_input
from tqdm import tqdm
from utils.fileutils import list_files_recursively, CacheStorage


class ImageNetPreprocessGenerator(ImageDataGenerator):
    def flow_from_directory(self,
                            directory,
                            target_size=(256, 256),
                            batch_size=32,
                            interpolation="nearest",
                            shuffle=True):
        self.batches = super(ImageNetPreprocessGenerator,
                             self).flow_from_directory(
                                 directory=directory,
                                 target_size=target_size,
                                 batch_size=batch_size,
                                 interpolation=interpolation,
                                 shuffle=shuffle,
                             )
        while True:
            batch_x, batch_y = next(self.batches)
            yield (preprocess_input(batch_x), batch_y)

    def reset(self):
        self.batches.reset()


def get_averaged_validation_image(directory):
    val_imgs, _ = get_validation_data(directory)  # preprocess済
    val_img_average = val_imgs.mean(axis=0)
    return val_img_average


def get_validation_dataflow(directory, batch_size=50):
    """validation_dataflow（preprocessを含むので後でもう一度使わないこと）
    Arguments:
        directory {[type]} -- [description]

    Keyword Arguments:
        batch_size {int} -- [description] (default: {50})

    Returns:
        [type] -- [description]
    """

    val_datagen = ImageNetPreprocessGenerator()
    val_dataflow = val_datagen.flow_from_directory(
        directory=str(directory),
        target_size=(224, 224),
        batch_size=batch_size,
        interpolation="bicubic",
        shuffle=False,
    )
    return val_dataflow


def load_imgs_through_dataflow(directory):
    val_dataflow = get_validation_dataflow(directory, batch_size=1)

    val_imgs = []
    val_labels = []

    num_imgs = 50000

    for i, (img, label) in enumerate(tqdm(val_dataflow, total=num_imgs)):
        val_imgs.append(img[0])
        val_labels.append(label[0])
        if i == (num_imgs - 1):
            break

    val_imgs = np.array(val_imgs)
    val_labels = np.array(val_labels)

    return val_imgs, val_labels


def get_saliency_maps(cache_directory):
    saliency_maps_dir = Path(cache_directory)

    cache = CacheStorage()
    print("Try to load cache file")
    saliency_maps = cache.get_cache(saliency_maps_dir / "saliency_maps")
    if saliency_maps is None:
        print("Making cache file")
        npz_path_list = list_files_recursively(saliency_maps_dir, ["npz"])
        saliency_maps = np.array(
            [np.load(npz_path)["x"] for npz_path in tqdm(npz_path_list)])
        cache.set_cache(saliency_maps_dir / "saliency_maps", saliency_maps)
    return saliency_maps


def get_validation_data(cache_directory):
    valid_data_dir = Path(cache_directory)
    val_imgs = []
    val_labels = []

    cache = CacheStorage()
    print("Try to load cache file")
    val_imgs = cache.get_cache(valid_data_dir / "val_imgs")
    val_labels = cache.get_cache(valid_data_dir / "val_labels")
    if val_imgs is None or val_labels is None:
        print("Making cache file")
        val_imgs, val_labels = load_imgs_through_dataflow(valid_data_dir)
        cache.set_cache(valid_data_dir / "val_imgs", val_imgs)
        cache.set_cache(valid_data_dir / "val_labels", val_labels)
    return val_imgs, val_labels


class PredictionKeeper(object):
    def __init__(self, predicts):
        self.predicts = predicts

        # pred_labels[5,0] <- 5+1番目の入力に対して最も自信のあるクラス
        self.predicted_labels = predicts.argsort()[:, ::-1]

    def get_index_pred_n(self, n, target_id):
        """尤度n番目の予測結果がtarget_idであるインデックスを取ってくる
        """
        return np.where(self.predicted_labels[:, n] == target_id)

    def get_all_nth_confidence_index(self, n):
        return self.predicted_labels[:, n]


def get_predkeeper(img_dir_path, modelpath=None):
    img_dir_path = Path(img_dir_path)
    cache_path = str((img_dir_path / "preds"))
    preds = None

    cache = CacheStorage()

    preds = cache.get_cache(cache_path)
    if preds is None:
        val_imgs, _ = get_validation_data(img_dir_path)
        model = VGG16(weights="imagenet") if modelpath is None else None
        preds = model.predict(val_imgs)
        cache.set_cache(cache_path, preds)
    predkeeper = PredictionKeeper(preds)
    return predkeeper


def evaluate_validation():
    valid_data_dir = Path("./data/ILSVRC2012_img_val_centercrop/")
    val_imgs, val_labels = get_validation_data(valid_data_dir)

    model = VGG19(weights='imagenet', )
    model.compile(
        loss='categorical_crossentropy',
        optimizer='SGD',
        metrics=["accuracy", "top_k_categorical_accuracy"],
    )
    print(model.metrics_names)
    print(model.evaluate(val_imgs, val_labels))


class AblateEvaluator(object):
    def __init__(self, model, averaged_X, patch_size=9):
        self.averaged_X = averaged_X  # averaged image through all dataset
        self.model = model
        self.rows = averaged_X.shape[0]
        self.cols = averaged_X.shape[1]
        self.patch_size = patch_size

    def pre_evaluate(self, X, Y, saliency_maps):
        assert X.shape[0:3] == saliency_maps.shape
        predictions_before_ablation = self.get_predictions_for_own_class(
            self.model, X, Y)
        print("before:", predictions_before_ablation.mean())
        
        return predictions_before_ablation
        
    def evaluate(self, X, Y, predictions_before_ablation, saliency_maps):
        assert X.shape[0:3] == saliency_maps.shape
        
        ablated_X = self.ablate(X, saliency_maps)

        predictions_after_ablation = self.get_predictions_for_own_class(
            self.model, ablated_X, Y)
        print("after:", predictions_after_ablation.mean())
        print("diff:", (predictions_before_ablation - predictions_after_ablation).mean())
        print("std:", (predictions_before_ablation - predictions_after_ablation).std())
        return (predictions_before_ablation - predictions_after_ablation).mean()

    def ablate(self, X, saliency_maps):
        """与えられた画像のうちsaliency mapの最高値を含むパッチについてデータセットの平均画像で置換

        Arguments:
            X {[type]} -- [description]
            saliency_maps {[type]} -- [description]

        Returns:
            [type] -- [description]
        """
        assert len(X.shape) == 4

        ablated_X = np.copy(X)

        for i, saliency_map in enumerate(saliency_maps):
            row, col = np.unravel_index(saliency_map.argmax(),
                                        saliency_map.shape)

            top = row - self.patch_size // 2
            bottom = row + self.patch_size // 2 + 1
            left = col - self.patch_size // 2
            right = col + self.patch_size // 2 + 1

            if top < 0:
                top = 0
            if bottom > self.rows - 1:
                bottom = self.rows - 1
            if left < 0:
                left = 0
            if right > self.cols - 1:
                right = self.cols - 1

            ablated_X[i, top:bottom, left:right, :] = self.averaged_X[
                top:bottom, left:right]

        return ablated_X

    @classmethod
    def get_predictions_for_own_class(cls, model, X, Y):
        return model.predict(X, verbose=1)[Y.astype(bool)]


class PointingGame(object):
    def __init__(self, masks):
        self.masks = masks
        self.num_samples = masks.shape[0]

    def calc_hitratio(self, binary_saliency_maps):
        assert self.masks.shape == binary_saliency_maps.shape
        num_fp_each_map = np.logical_and(
            self.masks, binary_saliency_maps).sum(axis=(1, 2))
        num_miss = np.count_nonzero(
            np.logical_or(
                num_fp_each_map, binary_saliency_maps.sum(axis=(1, 2)) == 0))
        hitratio = 1 - num_miss / self.num_samples
        num_miss2 = np.count_nonzero(
            np.logical_or(
                num_fp_each_map, binary_saliency_maps.sum(axis=(1, 2)) == 0), axis=1)
        hitratio2 = np.average(1 - num_miss2)
        std = np.std(1 - num_miss2)
        print(hitratio)
        print(num_miss2)
        print(std)
        return hitratio


def safe_divide(x, y):
    if y == 0:
        return 0
    else:
        return x / y


class PRevaluator(object):
    def __init__(self, masks):
        self.masks = masks.astype(bool)  # bboxがFalseのマスク
        self.num_real_positives = np.logical_not(self.masks).sum()
        self.num_real_negatives = self.masks.sum()
        self.tp = None

    def get_tp(self, binary_saliency_maps, masks=None):
        if masks is None:
            masks = self.masks
        return np.logical_and(np.logical_not(masks),
                              binary_saliency_maps).sum()

    def get_fp(self, binary_saliency_maps, masks=None):
        if masks is None:
            masks = self.masks
        return np.logical_and(masks, binary_saliency_maps).sum()

    def get_tn(self, binary_saliency_maps, masks=None):
        if masks is None:
            masks = self.masks
        return np.logical_and(masks,
                              np.logical_not(binary_saliency_maps)).sum()

    def get_fn(self, binary_saliency_maps, masks=None):
        if masks is None:
            masks = self.masks
        return np.logical_and(
            np.logical_not(masks), np.logical_not(binary_saliency_maps)).sum()

    def calc_precision(self, binary_saliency_maps):
        tp = self.get_tp(binary_saliency_maps)
        num_pred_positives = binary_saliency_maps.sum()
        precision = safe_divide(tp, num_pred_positives)
        return precision

    def calc_recall(self, binary_saliency_maps):
        tp = self.get_tp(binary_saliency_maps)
        recall = safe_divide(tp, self.num_real_positives)
        return recall

    def fmeasure(self, binary_saliency_maps):
        precision = self.calc_precision(binary_saliency_maps)
        recall = self.calc_recall(binary_saliency_maps)
        fvalue = 2 * precision * recall / (precision + recall)
        return fvalue

    def calc_fprate(self, binary_saliency_maps):
        fp = self.get_fp(binary_saliency_maps)
        fprate = fp / self.num_real_negatives
        return fprate

    def calc_tprate(self, binary_saliency_maps):
        tp = self.get_tp(binary_saliency_maps)
        tprate = tp / self.num_real_positives
        return tprate


class SaliencyBinarizerBase(object):
    def __init__(self, saliency_maps):
        self.norm_saliency_maps = saliency_maps

    def binarize(self, threshold):
        # threshold=1はすべてFalseになる
        binary_saliency_map = self.norm_saliency_maps >= threshold
        return binary_saliency_map

    def binarize_zero_or_not(self):
        binary_saliency_map = self.norm_saliency_maps > 0
        return binary_saliency_map

    def areadivide(self, remain_area_ratio):
        # 0より大きいピクセルの数を面積として保存
        # remain_area_ratioと面積をかけて残す面積の大きさを計算
        # np.zeros_like(saliency_map)
        # saliency_mapの値の大きい順に並べて残す面積分だけ1を代入
        pass


class MaxNormSaliencyBinarizer(SaliencyBinarizerBase):
    """尤度マップごとに最大値が1になるように正規化して，thを1-0で動かす
    """

    def __init__(self, saliency_maps):
        super(MaxNormSaliencyBinarizer, self).__init__(saliency_maps)
        max_each_map = saliency_maps.max(axis=(1, 2), keepdims=True)
        max_each_map[max_each_map == 0] = 1  # 1にしておくことで0除算を回避
        self.norm_saliency_maps /= max_each_map  # 各マップの最大値が1になるようにスケーリング


class MaxSaliencyBinarizer(SaliencyBinarizerBase):
    """尤度マップ全体で最大値が1になるように正規化して，thを1-0で動かす
    """

    def __init__(self, saliency_maps):
        super(MaxSaliencyBinarizer, self).__init__(saliency_maps)
        max_saliency = saliency_maps.max()
        if max_saliency == 0:
            max_saliency = 1  # 1にしておくことで0除算を回避
        self.norm_saliency_maps /= max_saliency  # 全マップの最大値が1になるようにスケーリング
