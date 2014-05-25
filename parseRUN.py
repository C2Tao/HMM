import numpy as np
from numpy.random import randint
from numpy.random import rand

from parseMLF import MLF
import urllib2
import pprint

A = MLF('/home/c2tao/Dropbox/Public/30_models/30_5034_50_3/result/result.mlf')
B = MLF('https://dl.dropboxusercontent.com/u/5029901/_readonly/vanilla_chn.mlf')

print B.tok_list

