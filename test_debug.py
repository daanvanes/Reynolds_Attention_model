# !/usr/bin/env python
# encoding: utf-8
"""
Created by Daan van Es on July 06 2016
Copyright (c) 2016 DE. All rights reserved.
"""

import numpy as np
import matplotlib.pyplot as plt
from IPython import embed as shell
from NormalizationModelofAttention import *

stimWidth = 5 
AxWidth = 30
Atheta = 0
AthetaWidth = 60

plotdir = os.path.join('/home/vanes/Reynolds_model/Figures')
if not os.path.isdir(plotdir):
	os.mkdir(plotdir)

NMA = NormalizationModelofAttention(plotdir)

# Sampling of space and orientation
x = np.mat(np.arange(-200,201))
theta = np.mat(np.arange(-180,181)).T
# Make stimuli
stimCenter1 = 100
stimOrientation1 = 0
stimCenter2 = -100
stimOrientation2 = 0
stim1 = NMA.make_gaussian(theta,stimOrientation1,1,1) * NMA.make_gaussian(x,stimCenter1,stimWidth,1) 
stim2 = NMA.make_gaussian(theta,stimOrientation2,1,1) * NMA.make_gaussian(x,stimCenter2,stimWidth,1)
stim = stim1 + stim2

# Attending stim 1
R1 = NMA.attention_model(x,theta,stim,Ax=stimCenter1,AxWidth=AxWidth,
  showActivityMaps=1,showModelParameters=1)
# # Attend orientation
# R2 = NMA.attention_model(x,theta,stim,Atheta=Atheta,AthetaWidth=AthetaWidth,
#   showActivityMaps=1,showModelParameters=1)
# # Attending stim 1 and orientation, oval
# R3 = NMA.attention_model(x,theta,stim,
#   Ax=stimCenter1,AxWidth=AxWidth,
#   Atheta=Atheta,AthetaWidth=AthetaWidth,
#   showActivityMaps=1,showModelParameters=1)
# # Attending stim 1 and orientation, cross
# R4 = NMA.attention_model(x,theta,stim,Ashape='cross',
#   Ax=stimCenter1,AxWidth=AxWidth,
#   Atheta=Atheta,AthetaWidth=AthetaWidth,
#   showActivityMaps=1,showModelParameters=1)




