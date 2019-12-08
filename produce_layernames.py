'''
Produces the Layer names of the model to help with finding and producing results
'''

import os
import sys
sys.path.append(".//src")
from model import *
from load import *
with open('.//params.json') as json_file:
    params = json.load(json_file)

model, modelwithlinear = loadmodel(modified = 0, source = params['sourcemodel'])
if(os.path.isfile("layer_name.txt")):
    os.remove("layer_name.txt")
file = open("layer_name.txt","w") 

for ilayer, layer in enumerate(model.layers):
    file.write("{:3.0f} {:10}".format(ilayer, layer.name)) 
    file.write("\n")
file.close()