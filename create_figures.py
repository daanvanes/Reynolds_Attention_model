# !/usr/bin/env python
# encoding: utf-8
"""
Created by Daan van Es on July 06 2016
Copyright (c) 2016 DE. All rights reserved.

"""

import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt
from IPython import embed as shell
import seaborn as sn
sn.set_style('ticks')
from figures import *
import os

res = 25
plotdir = os.path.join('/home/vanes/Reynolds_model/Figures')
if not os.path.isdir(plotdir):
	os.mkdir(plotdir)

figs = Figures(plotdir,res)


# uncomment figure to create below:
figs.Figure2A()
figs.Figure2B()
figs.Figure3C()
figs.Figure3F()
figs.Figure4C() 
figs.Figure4E() 
figs.Figure5C()
figs.Figure6C()
figs.Figure7C()
