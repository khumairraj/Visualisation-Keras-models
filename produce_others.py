'''
Codefile to produce the results for the following types: Saliency Map, Epsilon-LRP and Deeplift
'''
from vis.utils import utils
from vis.visualization import visualize_cam
from deepexplain.tensorflow import DeepExplain
from vis.visualization import visualize_cam, overlay

import os
import sys
import json
import tqdm
import numpy as np
import urllib.request
from matplotlib import pyplot as plt
from matplotlib.pyplot import figure
from skimage import feature, transform

import keras
import keras.backend as K
from keras.models import Model
from keras.datasets import mnist
from keras.utils import to_categorical
from keras.applications import MobileNetV2 as MOBILENET
from keras.preprocessing.image import load_img, img_to_array
from keras.applications.mobilenet_v2 import preprocess_input, decode_predictions

import warnings
warnings.filterwarnings("ignore")

sys.path.append(".\\src")
from model import *
from load import *

with open('params.json') as json_file:
    params = json.load(json_file)

if not os.path.exists(params["savepath"]):
    os.mkdir(params["savepath"])
savepath = params["savepath"]

classlabel = loadclasslabels(source = params['sourceclasslabel'])
model, modelwithlinear = loadmodel(modified = params['whethermodelmodified'], source = params['sourcemodel'], readtype = params["readtypeofmodel"])
imgs, _imglist, imgsname = loaddata(imgfolderpath = params['imgfolderpath'])
preds, class_idxs_sortedlist = runmodel(model, imgs)

topNclass = params["topNclass"]
with DeepExplain(session=K.get_session()) as de:
    input_tensor = model.layers[0].input
    fModel = Model(inputs=input_tensor, outputs = model.layers[-1].output)
    target_tensor = fModel(input_tensor)

    for u, img in tqdm.tqdm(zip(range(imgs.shape[0]), imgs)):
        labelstostudy = class_idxs_sortedlist[u][:topNclass]

        ys = to_categorical(labelstostudy, num_classes=1000) # one-hot encode the predicted indices
        xs = np.tile(imgs[u], (len(labelstostudy), 1, 1, 1)) # Duplicate the image N_PLOT_PRED number of times

        # Draw saliency maps and Epsilon-LRP heatmap
        attributions = {
            'Saliency maps': de.explain('saliency', fModel.outputs[0] * ys, fModel.inputs[0], xs),
            'Epsilon-LRP': de.explain('elrp', fModel.outputs[0] * ys, fModel.inputs[0], xs),
            'DeepLift': de.explain('deeplift', fModel.outputs[0] * ys, fModel.inputs[0], xs)
        }

        # Plotting Function
        n_cols = int(len(attributions)) + 1
        n_rows = len(labelstostudy)
        fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(5*n_cols, 5*n_rows))
        
        for i, xi in enumerate(xs):
            print('{}: {:.2f}%'.format(classlabel[labelstostudy[i]], preds[u][labelstostudy[i]] * 100))

            xi = (xi - np.min(xi))
            xi /= np.max(xi)
            ax = axes[i][0]
            ax.imshow(_imglist[u])
            ax.set_title('Original')
            ax.axis('off')

            for j, a in enumerate(attributions):
                axj = axes[i][j + 1]
                plot_custom(attributions[a][i], xi = xi, axis=axj, dilation=.5, percentile=99, alpha=.2).set_title(a)
        plt.savefig(os.path.join(savepath, imgsname[u]))