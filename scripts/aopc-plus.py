import numpy as np
from tqdm import tqdm
from keras.applications.vgg19 import VGG19, preprocess_input, decode_predictions
from keras.models import Model
from utils.helper import heatmap
from innvestigate import utils as iutils
import os
import sys
import itertools
import argparse

from pathlib import Path
from utils.fileutils import CacheStorage
from utils.evaluations import get_validation_data

def define_mask(row, col, input_shape, windowsize=9):
    top = row - windowsize // 2
    bottom = row + windowsize // 2 + 1
    left = col - windowsize // 2
    right = col + windowsize // 2 + 1

    if top < 0:
        top = 0
    if bottom > input_shape[1] - 1:
        bottom = input_shape[1] - 1
    if left < 0:
        left = 0
    if right > input_shape[2] - 1:
        right = input_shape[2] - 1
    return top, bottom, left, right

def run_change_in_y_t(model, targets, images, maps, occlusion, predictions, input_shape=(50000, 224,224,3), num_iters=100, windowsize=9, mask="pixel"):
    ret = np.zeros((input_shape[0], num_iters))

    inf_map = np.full_like(maps[0], -np.inf)
    for example_id, target_id in enumerate(tqdm(targets)):
        modified_images = np.zeros((num_iters, input_shape[1], input_shape[2], input_shape[3]))
        modified_images[0] = np.copy(images[example_id])
        current_map = np.copy(maps[example_id])
        for i in np.arange(1,num_iters):
            # cant handle same max
#             row, col = np.unravel_index(current_map.argmax(), current_map.shape)
            row, col = np.unravel_index(np.random.choice(np.flatnonzero(current_map == current_map.max())), current_map.shape)

            top, bottom, left, right = define_mask(row, col, input_shape, windowsize)
            modified_images[i] = modified_images[i-1]
            modified_images[i, top:bottom, left:right] = occlusion[top:bottom, left:right]
            
            # wipe out used
            if mask == "box":
                current_map[top:bottom, left:right] = inf_map[top:bottom, left:right]
            else:
                current_map[row, col] = -np.inf
        ret[example_id] = model.predict(modified_images)[:,target_id]
    return ret

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AOPC calculator")
    parser.add_arguement('-g', '--gpu', type=str, default="0", help='Controls which gpus cuda uses')
    parser.add_arguement('-m', '--method', type=str, default="random", help='The method to run')
    parser.add_arguement('-n', '--network', type=str, default="vgg19", help='The network model')
    parser.add_arguement('-i', '--iters', type=int, default=100, help='How many iterations')
    parser.add_arguement('-w', '--windowsize', type=int, default=9, help='Window size')
    parser.add_arguement('-k', '--mask', type=int, default="pixel", help='Mask, "box" or "pixel"')
    args = parser.parse_args()
    
    os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu
    method = args.method
    network = args.network
    num_iters = args.iters
    windowsize = args.windowsize
    mask = args.mask
    
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
        predictions = model.predict(val_imgs)
        cache.set_cache(os.path.join('cache', "imagenet_predictions"), predictions)
        
    # load maps
    print("Loading Maps")
    if method != "random":
        npz_path_list = [os.path.join(store_dir, method, "%d_%d_%s_%s_gt.npz"%(c, i, network, method)) for c, i in itertools.product(range(num_classes), range(num_per_class))]
        orig_maps = np.array([np.load(npz_path)["x"] for npz_path in tqdm(npz_path_list)])
    else:
        orig_maps = np.random.uniform(0, 255, (num_classes*num_per_class, input_shape[1], input_shape[2]))
    
    # run results
    results = run_change_in_y_t(model, gt, val_imgs, orig_maps, occlusion, predictions, input_shape=input_shape, num_iters=num_iters, windowsize=windowsize, mask=mask)
    
    results_path = os.path.join("output", "full_fullresults_plus_imagenet_%s_%s_gt_%d_w%d_%s.npz"%(network, method, num_iters, windowsize, mask))
    np.savez_compressed(os.path.join(results_path), full=results)
            
    # calc ave
    ave_results = np.mean(results, axis=0)
    
    # calc aopc
    f_orig = np.full(num_iters, ave_results[0])
    aopc = 1./(num_iters+1.)*np.cumsum(f_orig - ave_results)

    # save results
    results_path = os.path.join("output", "full_ave_aopc_plus_imagenet_%s_%s_gt_%d_w%d_%s.npz"%(network, method, num_iters, windowsize, mask))
    np.savez_compressed(os.path.join(results_path), ave=ave_results, aopc=aopc)
