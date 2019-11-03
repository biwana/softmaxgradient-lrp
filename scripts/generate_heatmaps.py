import numpy as np
from tqdm import tqdm
from pathlib import Path
from keras.applications.vgg19 import VGG19, preprocess_input, decode_predictions
from keras import backend as K
from keras.models import Model
from innvestigate import utils as iutils
import os
import sys
from keras.preprocessing.image import img_to_array, load_img

from utils.evaluations import get_validation_data
from utils.visualizations import GradCAM, GuidedGradCAM, GBP, LRP, CLRP, SGLRP

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"]=sys.argv[1]
    
    print("load data")
    valid_data_dir = Path("./data/ILSVRC2012_val/img_centercrop/")
    val_imgs, val_labels = get_validation_data(valid_data_dir)
    min_input = np.min(val_imgs)
    max_input = np.max(val_imgs)
    input_shape = val_imgs.shape
    
    start_class = int(sys.argv[2])
    end_class = int(sys.argv[3])

    classes = np.arange(start_class, end_class)
    num_per_class = 50
    
    use_relu = False
    
    store_dir = os.path.join("maps", "imagenet")
    
    target_layer = "block5_pool"
    

    
    for c in tqdm(classes):
        start = c * num_per_class
        end = start + num_per_class
        images = val_imgs[start:end]

        model = VGG19(weights="imagenet")
        partial_model = Model(
            inputs=model.inputs,
            outputs=iutils.keras.graph.pre_softmax_tensors(model.outputs),
            name=model.name,
        )
        
        network = "vgg19"

        if not os.path.isfile(os.path.join(store_dir, "%d_%d_%s_lrp.npz"%(c, num_per_class-1, network))):
            analyzer = LRP(
                partial_model,
                target_id=c,
                relu=use_relu,
                low=min_input,
                high=max_input,
            )
            analysis = analyzer.analyze(images)
            for i in range(num_per_class):
                if not os.path.isfile(os.path.join(store_dir, "%d_%d_%s_lrp.npz"%(c, i, network))):
                    np.savez_compressed(os.path.join(store_dir, "%d_%d_%s_lrp"%(c, i, network)), x=analysis[i].sum(axis=(2)))

        if not os.path.isfile(os.path.join(store_dir, "%d_%d_%s_sglrp.npz"%(c, num_per_class-1, network))):
            analyzer = SGLRP(
                partial_model,
                target_id=c,
                relu=use_relu,
                low=min_input,
                high=max_input,
            )
            analysis = analyzer.analyze(images)
            for i in range(num_per_class):
                if not os.path.isfile(os.path.join(store_dir, "%d_%d_%s_sglrp.npz"%(c, i, network))):
                    np.savez_compressed(os.path.join(store_dir, "%d_%d_%s_sglrp"%(c, i, network)), x=analysis[i].sum(axis=(2)))



        if not os.path.isfile(os.path.join(store_dir, "%d_%d_%s_clrp.npz"%(c, num_per_class-1, network))):
            analyzer = CLRP(
                partial_model,
                target_id=c,
                relu=use_relu,
                low=min_input,
                high=max_input,
            )
            analysis = analyzer.analyze(images)
            for i in range(num_per_class):
                if not os.path.isfile(os.path.join(store_dir, "%d_%d_%s_clrp.npz"%(c, i, network))):
                    np.savez_compressed(os.path.join(store_dir, "%d_%d_%s_clrp"%(c, i, network)), x=analysis[i].sum(axis=(2)))

        if not os.path.isfile(os.path.join(store_dir, "%d_%d_%s_gbp.npz"%(c, num_per_class-1, network))):
            analyzer = GBP(
                partial_model,
                target_id=c,
                relu=use_relu,
            )
            analysis = analyzer.analyze(images)
            for i in range(num_per_class):
                if not os.path.isfile(os.path.join(store_dir, "%d_%d_%s_gbp.npz"%(c, i, network))):
                    np.savez_compressed(os.path.join(store_dir, "%d_%d_%s_gbp"%(c, i, network)), x=analysis[i].sum(axis=(2)))


        if not os.path.isfile(os.path.join(store_dir, "%d_%d_%s_ggc.npz"%(c, num_per_class-1, network))):
            analyzer = GuidedGradCAM(
                partial_model,
                target_id=c,
                layer_name=target_layer,
                relu=use_relu,
            )
            analysis = analyzer.analyze(images)
            for i in range(num_per_class):
                if not os.path.isfile(os.path.join(store_dir, "%d_%d_%s_ggc.npz"%(c, i, network))):
                    np.savez_compressed(os.path.join(store_dir, "%d_%d_%s_ggc"%(c, i, network)), x=analysis[i].sum(axis=(2)))
        K.clear_session()