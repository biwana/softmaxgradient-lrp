import numpy as np
from tqdm import tqdm
from keras.applications.vgg19 import VGG19, preprocess_input, decode_predictions
from keras.models import Model
from utils.helper import heatmap
from innvestigate import utils as iutils
import os
import sys
import itertools


from pathlib import Path
from utils.fileutils import CacheStorage
from utils.evaluations import get_validation_data

def define_mask(row, col, input_shape):
    top = row - 9 // 2
    bottom = row + 9 // 2 + 1
    left = col - 9 // 2
    right = col + 9 // 2 + 1

    if top < 0:
        top = 0
    if bottom > input_shape[1] - 1:
        bottom = input_shape[1] - 1
    if left < 0:
        left = 0
    if right > input_shape[2] - 1:
        right = input_shape[2] - 1
    return top, bottom, left, right

def run_change_in_y_t(model, targets, images, maps, occlusion, predictions, input_shape=(50000, 224,224,3), num_iters=100):
    ret = np.zeros((input_shape[0], num_iters))

    for example_id, target_id in enumerate(tqdm(targets)):
        modified_images = np.zeros((num_iters, input_shape[1], input_shape[2], input_shape[3]))
        modified_images[0] = np.copy(images[example_id])
        current_map = np.copy(maps[example_id])
        for i in np.arange(1,num_iters):
            row, col = np.unravel_index(current_map.argmax(), current_map.shape)

            # wipe out used
            current_map[row, col] = 0

            top, bottom, left, right = define_mask(row, col, input_shape)
            modified_images[i] = modified_images[i-1]
            modified_images[i, top:bottom, left:right] = occlusion[top:bottom, left:right]
        ret[example_id] = model.predict(modified_images)[:,target_id]
    return ret

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"]=sys.argv[1]
    method = sys.argv[2]
    network = sys.argv[3]
    
    num_iters = 100
    num_classes = 1000
    num_per_class = 50
    store_dir = os.path.join("maps", "imagenet")
    
    # load data
    valid_data_dir = Path("./data/ILSVRC2012_val/img_centercrop/")
    val_imgs, val_labels = get_validation_data(valid_data_dir)
    gt = np.argmax(val_labels, axis=1)
    input_shape = val_imgs.shape
    
    # load model
    model = VGG19(weights="imagenet")
    model.compile(
        loss='categorical_crossentropy',
        optimizer='SGD',
        metrics=["accuracy", "top_k_categorical_accuracy"],
    )
    partial_model = Model(
        inputs=model.inputs,
        outputs=iutils.keras.graph.pre_softmax_tensors(model.outputs),
        name=model.name,
    )
    
    # define occlusion
    occlusion = np.random.uniform(0, 255, (input_shape[1], input_shape[2], input_shape[3]))
    occlusion = preprocess_input(occlusion)
    
    # load predictions
    cache = CacheStorage()
    print("Try to load cache file")
    predictions = cache.get_cache(os.path.join('cache', "imagenet_predictions"))
    if predictions is None:
        print("Making cache file")
        predictions = model.predict(input_imgs)
        cache.set_cache(os.path.join('cache', "imagenet_predictions"), predictions)
        
    # load maps
    print("Loading Maps")
    npz_path_list = [os.path.join(store_dir, "%d_%d_%s_%s.npz"%(c, i, network, method)) for c, i in itertools.product(range(num_classes), range(num_per_class))]
    orig_maps = np.array([np.load(npz_path)["x"] for npz_path in tqdm(npz_path_list)])
    
    # run results
    results = run_change_in_y_t(model, gt, val_imgs, orig_maps, occlusion, predictions, input_shape=input_shape, num_iters=100)
    
        
    # calc ave
    ave_results = np.mean(results, axis=0)
    
    # calc aopc
    f_orig = np.full(num_iters, ave_results[0])
    aopc = 1./(num_iters+1.)*np.cumsum(f_orig - ave_results)

    # save results
    results_path = os.path.join("output", "full_results_imagenet_%s_%s_gt.npz"%(network, method))
    np.savez_compressed(os.path.join("output", results_path), full=results, ave=ave_results, aopc=aopc)
