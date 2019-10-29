from common2.util import read_files_from_dir
from common2 import util
import re
import numpy as np
import pickle
import matplotlib.pyplot as plt
import os

#regex = re.compile(r'^dev')
lista = []
#lista = read_files_from_dir('../data/', r'^dev')
#lista = util.process_htk_files_for_fishers_normal('../data/', r'^train')
#for entry in lista:
 #   print("esto ", entry)
train = '../audio/timit/TRAIN'
test = '../audio/timit/TEST'

wavs = []
for root, dirs, files in os.walk(train):
    for file in files:
        if file.endswith(".WAV"):
             wavs.append(os.path.join(root, file))
