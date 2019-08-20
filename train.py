import sys
import matplotlib.pyplot as plt
import numpy as np
import pickle
import glob
from music21 import converter, instrument, note, chord, stream
from utils import *
from GAN import GAN

gan = GAN(rows=100)    
gan.train(epochs=5000, batch_size=32, sample_interval=1)
