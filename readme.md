# Understanding the output of MobileNetv2
### Outline of the Process

  - The GCAM method helps to analyse which part the model is focussing while making a prediction
  - The other method include the following: Saliency Map, Epsilon-LRP and Deeplift. These give 
    an idea of the saliency of the model
  - The model is working on mobilenetv2 now. Provisions are made to analyse any model.

### Analyse Results
#### Layer view analysis
```
The initial layer focus on textures and features
```
![Block_2_expand](/readmefiles/catdog3_block_2_expand.jpg)
```
As we move forward in deep layers the focus starts shifting
```
![Block_9_expand](/readmefiles/catdog3_block_9_expand.jpg)
```
Finally it starts seeing high level features in the image. It is to observe how the model sees at different location for 
- dog and 
- lion
```
![Conv_1](/readmefiles/catdog3_Conv_1.jpg)

#### Saliency Analysis
```
This are the salient features in the input image
```
![Saliency](/readmefiles/catdog3.jpg)
### Requirements
- The code has been written for python 3.6
- Run the following command
```
    pip install -r requirements.txt
```
- Run the following command to produce the layernames which can be used in the params. Default is set though.
```
    python produce_layernames.py
```
- Run the following command to produce the GCAM outputs
```
    python produce_GCAM.py
```
- Run the following command to produce the other outputs
```
    python produce_others.py
```

### Parameters
The params contain the following keys
```
   "sourceclasslabel" : default = "imagenet" else provide the path for the classnames.json
```
```
   "sourcemodel": 
                        default = "imagenet". Else change it to None and save the model in
                        the model folder.
```
```
    "readtypeofmodel" : 
                        default : "plain". Specify the type in which model is saved. Other 
                        options are : "json" and "yaml". Please make sure the model is 
                        saved in this format in the model folder.
```
```
    "whethermodelmodified" : 
                        default = 0. Specify if the model has been modified before the
                        previous run. This is to reduce the time spent in removing the
                        softmax in final layer
```
```
    "imgfolderpath":    default = ".\\sample"
```
```
    "savepath" : 		default = ".\\analyse_results"
```
```
    "penultimate_layer_names" :
                        default = ["block_2_expand", "block_9_expand", "Conv_1"]
                        Specify the layers to study from the network. To get the layernames
                        run the file producelayernames.py
```
```
    "topNclass" :       default = 2. Set the number of outputs of the model to study.
```