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
from NormalizationModelofAttention import *

class Figures(object):

	def __init__(self,):

		self.numContrasts = 25
		self.numOrientations = 25

	def Figure2A(self,):

		NMA = NormalizationModelofAttention()

		titleString = 'Figure 2A: large att, small stim'
		stimWidth = 3 
		AxWidth = 30
		cRange = [1e-5, 1]

		# Sampling of space and orientation
		x =  np.mat(np.arange(-200,201))
		theta = np.mat(np.arange(-180,181)).T

		# Make stimuli
		stimCenter1 = 100
		stimOrientation1 = 0
		stimCenter2 = -100
		stimOrientation2 = 0
		stim1 = NMA.make_gaussian(theta,stimOrientation1,1,1) * NMA.make_gaussian(x,stimCenter1,stimWidth,1) 
		stim2 = NMA.make_gaussian(theta,stimOrientation2,1,1) * NMA.make_gaussian(x,stimCenter2,stimWidth,1)

		# Pick contrasts
		logCRange = np.log10(cRange)
		logContrasts = np.linspace(	logCRange[0],logCRange[1],self.numContrasts)
		contrasts = 10**logContrasts

		# Pick neuron to record
		j = np.where(np.ravel(theta)==stimOrientation1)[0][0]
		i = np.where(np.ravel(x)==stimCenter1)[0][0]

		attCRF = np.zeros(np.size(contrasts))
		unattCRF = np.zeros(np.size(contrasts))
		for c in range(0,self.numContrasts):
			stim = contrasts[c] * stim1 + contrasts[c] * stim2
			if c == (self.numContrasts-1):
				showActivityMaps = 1
				showModelParameters = 1
			else:
				showActivityMaps = 0
				showModelParameters = 0
			# Population response when attending stim 1
			R1 = NMA.attention_model(x,theta,stim,Ax=stimCenter1,AxWidth=AxWidth,
				showActivityMaps=showActivityMaps,showModelParameters=showModelParameters)
			# Population response when attending stim 2
			R2 = NMA.attention_model(x,theta,stim,Ax=stimCenter2,AxWidth=AxWidth)
			attCRF[c] = R1[j,i]
			unattCRF[c] = R2[j,i]

		f = plt.figure(figsize=(7,7))
		s = f.add_subplot(121)
		plt.plot(contrasts,unattCRF)
		plt.plot(contrasts,attCRF)
		s.set_xscale('log')
		plt.ylim([0, 30])
		plt.xlim(cRange)
		plt.legend(['Att Away','Att RF'],loc='best')
		plt.ylabel('Normalized response')
		plt.xlabel('Log contrast')
		plt.title(titleString)
		sn.despine(offset=10)
		s = f.add_subplot(122)
		plt.plot(contrasts,100*(attCRF-unattCRF)/unattCRF)
		s.set_xscale('log')
		plt.ylim([0, 100])
		plt.xlim(cRange)
		plt.ylabel('Attentional modulation (%)')
		plt.xlabel('Log contrast')
		sn.despine(offset=10)
		plt.tight_layout()


	def Figure2B(self,):

		NMA = NormalizationModelofAttention()

		titleString = 'Figure 2B: small att, large stim'
		stimWidth = 5 
		AxWidth = 3
		cRange = [1e-5, 1]

		# Sampling of space and orientation
		x =  np.mat(np.arange(-200,201))
		theta = np.mat(np.arange(-180,181)).T

		# Make stimuli
		stimCenter1 = 100
		stimOrientation1 = 0
		stimCenter2 = -100
		stimOrientation2 = 0
		stim1 = NMA.make_gaussian(theta,stimOrientation1,1,1) * NMA.make_gaussian(x,stimCenter1,stimWidth,1) 
		stim2 = NMA.make_gaussian(theta,stimOrientation2,1,1) * NMA.make_gaussian(x,stimCenter2,stimWidth,1)

		# Pick contrasts
		logCRange = np.log10(cRange)
		logContrasts = np.linspace(	logCRange[0],logCRange[1],self.numContrasts)
		contrasts = 10**logContrasts

		# Pick neuron to record
		j = np.where(np.ravel(theta)==stimOrientation1)[0][0]
		i = np.where(np.ravel(x)==stimCenter1)[0][0]

		attCRF = np.zeros(np.size(contrasts))
		unattCRF = np.zeros(np.size(contrasts))

		for c in range(0,self.numContrasts):
			stim = contrasts[c] * stim1 + contrasts[c] * stim2
			# Population response when attending stim 1
			R1 = NMA.attention_model(x,theta,stim,Ax=stimCenter1,AxWidth=AxWidth)
			# Population response when attending stim 2
			R2 = NMA.attention_model(x,theta,stim,Ax=stimCenter2,AxWidth=AxWidth)
			attCRF[c] = R1[j,i]
			unattCRF[c] = R2[j,i]

		f = plt.figure(figsize=(7,7))
		s = f.add_subplot(121)
		plt.plot(contrasts,unattCRF)
		plt.plot(contrasts,attCRF)
		s.set_xscale('log')
		plt.ylim([0, 30])
		plt.xlim(cRange)
		plt.legend(['Att Away','Att RF'],loc='best')
		plt.ylabel('Normalized response')
		plt.xlabel('Log contrast')
		plt.title(titleString)
		sn.despine(offset=10)
		s = f.add_subplot(122)
		plt.plot(contrasts,100*(attCRF-unattCRF)/unattCRF)
		s.set_xscale('log')
		plt.ylim([0, 100])
		plt.xlim(cRange)
		plt.ylabel('Attentional modulation (%)')
		plt.xlabel('Log contrast')
		sn.despine(offset=10)
		plt.tight_layout()


	def Figure3C(self,):

		NMA = NormalizationModelofAttention()

		titleString = 'Figure 3C (Reynolds, Pasternak & Desimone, 2000)'
		stimWidth = 5 
		AxWidth = 30
		baselineMod = 5e-7 
		baselineUnmod = 5
		cRange = [1e-5, 1]

		# Sampling of space and orientation
		x =  np.mat(np.arange(-200,201))
		theta = np.mat(np.arange(-180,181)).T

		# Make stimuli
		stimCenter1 = 100
		stimOrientation1 = 0
		stimCenter2 = -100
		stimOrientation2 = 0
		stim1 = NMA.make_gaussian(theta,stimOrientation1,1,1) * NMA.make_gaussian(x,stimCenter1,stimWidth,1) 
		stim2 = NMA.make_gaussian(theta,stimOrientation2,1,1) * NMA.make_gaussian(x,stimCenter2,stimWidth,1)

		# Pick contrasts
		logCRange = np.log10(cRange)
		logContrasts = np.linspace(	logCRange[0],logCRange[1],self.numContrasts)
		contrasts = 10**logContrasts

		# Pick neuron to record
		j = np.where(np.ravel(theta)==stimOrientation1)[0][0]
		i = np.where(np.ravel(x)==stimCenter1)[0][0]

		attCRF = np.zeros(np.size(contrasts))
		unattCRF = np.zeros(np.size(contrasts))

		for c in range(0,self.numContrasts):
			stim = contrasts[c] * stim1 + contrasts[c] * stim2
			# Population response when attending stim 1
			R1 = NMA.attention_model(x,theta,stim,Ax=stimCenter1,AxWidth=AxWidth,
				baselineMod=baselineMod,baselineUnmod=baselineUnmod)
			# Population response when attending stim 2
			R2 = NMA.attention_model(x,theta,stim,Ax=stimCenter2,AxWidth=AxWidth,
				baselineMod=baselineMod,baselineUnmod=baselineUnmod)
			attCRF[c] = R1[j,i]
			unattCRF[c] = R2[j,i]

		f = plt.figure(figsize=(7,7))
		s = f.add_subplot(121)
		plt.plot(contrasts,unattCRF)
		plt.plot(contrasts,attCRF)
		s.set_xscale('log')
		plt.ylim([0, 25])
		plt.xlim(cRange)
		plt.legend(['Att Away','Att RF'],loc='best')
		plt.ylabel('Normalized response')
		plt.xlabel('Log contrast')
		plt.title(titleString)
		sn.despine(offset=10)
		s = f.add_subplot(122)
		plt.plot(contrasts,100*(attCRF-unattCRF)/unattCRF)
		s.set_xscale('log')
		plt.ylim([0, 100])
		plt.xlim(cRange)
		plt.ylabel('Attentional modulation (%)')
		plt.xlabel('Log contrast')
		sn.despine(offset=10)
		plt.tight_layout()


	def Figure3F(self,):

		NMA = NormalizationModelofAttention()

		titleString = 'Figure 3F (Williford & Maunsell, 2007)'
		stimWidth = 7 
		AxWidth = 7
		baselineMod = 5e-7 
		baselineUnmod = 0
		cRange = [1e-5, 1]

		# Sampling of space and orientation
		x =  np.mat(np.arange(-200,201))
		theta = np.mat(np.arange(-180,181)).T

		# Make stimuli
		stimCenter1 = 100
		stimOrientation1 = 0
		stimCenter2 = -100
		stimOrientation2 = 0
		stim1 = NMA.make_gaussian(theta,stimOrientation1,1,1) * NMA.make_gaussian(x,stimCenter1,stimWidth,1) 
		stim2 = NMA.make_gaussian(theta,stimOrientation2,1,1) * NMA.make_gaussian(x,stimCenter2,stimWidth,1)

		# Pick contrasts
		logCRange = np.log10(cRange)
		logContrasts = np.linspace(	logCRange[0],logCRange[1],self.numContrasts)
		contrasts = 10**logContrasts

		# Pick neuron to record
		j = np.where(np.ravel(theta)==stimOrientation1)[0][0]
		i = np.where(np.ravel(x)==stimCenter1)[0][0]

		attCRF = np.zeros(np.size(contrasts))
		unattCRF = np.zeros(np.size(contrasts))

		for c in range(0,self.numContrasts):
			stim = contrasts[c] * stim1 + contrasts[c] * stim2
			# Population response when attending stim 1
			R1 = NMA.attention_model(x,theta,stim,Ax=stimCenter1,AxWidth=AxWidth,
				baselineMod=baselineMod,baselineUnmod=baselineUnmod)
			# Population response when attending stim 2
			R2 = NMA.attention_model(x,theta,stim,Ax=stimCenter2,AxWidth=AxWidth,
				baselineMod=baselineMod,baselineUnmod=baselineUnmod)
			attCRF[c] = R1[j,i]
			unattCRF[c] = R2[j,i]

		f = plt.figure(figsize=(7,7))
		s = f.add_subplot(121)
		plt.plot(contrasts,unattCRF)
		plt.plot(contrasts,attCRF)
		s.set_xscale('log')
		plt.ylim([0, 25])
		plt.xlim(cRange)
		plt.legend(['Att Away','Att RF'],loc='best')
		plt.ylabel('Normalized response')
		plt.xlabel('Log contrast')
		plt.title(titleString)
		sn.despine(offset=10)
		s = f.add_subplot(122)
		plt.plot(contrasts,100*(attCRF-unattCRF)/unattCRF)
		s.set_xscale('log')
		plt.ylim([0, 100])
		plt.xlim(cRange)
		plt.ylabel('Attentional modulation (%)')
		plt.xlabel('Log contrast')
		sn.despine(offset=10)
		plt.tight_layout()


	def Figure4C(self,):

		NMA = NormalizationModelofAttention()

		titleString = 'Figure 4C (Martinez-Trujillo & Treue, 2002)'
		stimWidth = 5 
		AxWidth = 5
		AthetaWidth = 20
		Apeak = 5
		cRange = [1e-4, 0.1]

		# Sampling of space and orientation
		x =  np.mat(np.arange(-200,201))
		theta = np.mat(np.arange(-180,181)).T

		# Make stimuli
		stimCenter1 = 90
		stimOrientation1 = 0
		stimCenter2 = 110
		stimOrientation2 = 180
		stimCenter3 = -90
		stimOrientation3 = 0
		stimCenter4 = -110
		stimOrientation4 = 180

		# The contrast of the null stimulus (fixed in contrast).
		fixed_contrast = .01

		# Choose neuron with RF centered at midpoint between the two stim intended
		# to be in RF:
		RF_center = round(np.mean([stimCenter1, stimCenter2]))

		# Stim 1 and 2 in RF
		stim1 = NMA.make_gaussian(theta,stimOrientation1,1,1) * NMA.make_gaussian(x,stimCenter1,stimWidth,1)
		stim2 = NMA.make_gaussian(theta,stimOrientation2,1,1) * NMA.make_gaussian(x,stimCenter2,stimWidth,1)

		# Stim 3 and 4 contralateral to RF
		stim3 = NMA.make_gaussian(theta,stimOrientation1,1,1) * NMA.make_gaussian(x,stimCenter3,stimWidth,1)
		stim4 = NMA.make_gaussian(theta,stimOrientation2,1,1) * NMA.make_gaussian(x,stimCenter4,stimWidth,1)

		# Pick contrasts
		logCRange = np.log10(cRange)
		logContrasts = np.linspace(	logCRange[0],logCRange[1],self.numContrasts)
		contrasts = 10**logContrasts

		# Pick neuron to record
		j = np.where(np.ravel(theta)==stimOrientation1)[0][0]
		i = np.where(np.ravel(x)==stimCenter1)[0][0]

		attCRF = np.zeros(np.size(contrasts))
		unattCRF = np.zeros(np.size(contrasts))
		for c in range(0,self.numContrasts):
			stim = contrasts[c] * stim1 + fixed_contrast * stim2 + contrasts[c] * stim3 + fixed_contrast * stim4
			# Population response when attending null stim in RF:
			R1 = NMA.attention_model(x,theta,stim,Apeak=Apeak,
				Ax=stimCenter2,AxWidth=AxWidth,
				Atheta=stimOrientation2,AthetaWidth=AthetaWidth,
				showActivityMaps=0,showModelParameters=0)
			# Population response when attending null stim contralateral to RF:
			R2 = NMA.attention_model(x,theta,stim,Apeak=Apeak,
				Ax=stimCenter4,AxWidth=AxWidth,
				Atheta=stimOrientation2,AthetaWidth=AthetaWidth,
				showActivityMaps=0,showModelParameters=0)
			attCRF[c] = R1[j,i]
			unattCRF[c] = R2[j,i]

		f = plt.figure(figsize=(7,7))
		s = f.add_subplot(121)
		plt.plot(contrasts,unattCRF)
		plt.plot(contrasts,attCRF)
		s.set_xscale('log')
		plt.ylim([0, 7])
		plt.xlim(cRange)
		plt.legend(['Att Away','Att RF'],loc='best')
		plt.ylabel('Normalized response')
		plt.xlabel('Log contrast')
		plt.title(titleString)
		sn.despine(offset=10)
		s = f.add_subplot(122)
		plt.plot(contrasts,100*(unattCRF-attCRF)/unattCRF)
		s.set_xscale('log')
		plt.ylim([0, 100])
		plt.xlim(cRange)
		plt.ylabel('Attentional modulation (%)')
		plt.xlabel('Log contrast')
		sn.despine(offset=10)
		plt.tight_layout()


	def Figure4E(self,):

		NMA = NormalizationModelofAttention()

		titleString = 'Figure 4C (Martinez-Trujillo & Treue, 2002)'
		stimWidth = 5 
		AxWidth = 5
		AthetaWidth = 20
		Apeak = 5
		cRange = [1e-4, 0.1]

		# Sampling of space and orientation
		x =  np.mat(np.arange(-200,201))
		theta = np.mat(np.arange(-180,181)).T

		# Make stimuli
		stimCenter1 = 90
		stimOrientation1 = 0
		stimCenter2 = 110
		stimOrientation2 = 180
		stimCenter3 = -90
		stimOrientation3 = 0
		stimCenter4 = -110
		stimOrientation4 = 180

		# Choose neuron with RF centered at midpoint between the two stim intended
		# to be in RF:
		RF_center = round(np.mean([stimCenter1, stimCenter2]))

		# Stim 1 and 2 in RF
		stim1 = NMA.make_gaussian(theta,stimOrientation1,1,1) * NMA.make_gaussian(x,stimCenter1,stimWidth,1)
		stim2 = NMA.make_gaussian(theta,stimOrientation2,1,1) * NMA.make_gaussian(x,stimCenter2,stimWidth,1)

		# Stim 3 and 4 contralateral to RF
		stim3 = NMA.make_gaussian(theta,stimOrientation1,1,1) * NMA.make_gaussian(x,stimCenter3,stimWidth,1)
		stim4 = NMA.make_gaussian(theta,stimOrientation2,1,1) * NMA.make_gaussian(x,stimCenter4,stimWidth,1)

		# Pick contrasts
		logCRange = np.log10(cRange)
		logContrasts = np.linspace(	logCRange[0],logCRange[1],self.numContrasts)
		contrasts = 10**logContrasts

		# Pick neuron to record
		j = np.where(np.ravel(theta)==stimOrientation1)[0][0]
		i = np.where(np.ravel(x)==stimCenter1)[0][0]

		attCRF = np.zeros(np.size(contrasts))
		unattCRF = np.zeros(np.size(contrasts))
		for c in range(0,self.numContrasts):
			stim = contrasts[c] * stim1 + contrasts[c] * stim2 + contrasts[c] * stim3 + contrasts[c] * stim4
			# Population response when attending preferred stim in RF:
			R1 = NMA.attention_model(x,theta,stim,Apeak=Apeak,
				Ax=stimCenter1,AxWidth=AxWidth,
				Atheta=stimOrientation1,AthetaWidth=AthetaWidth,
				showActivityMaps=0,showModelParameters=0)
			# Population response when attending null stim in RF:
			R2 = NMA.attention_model(x,theta,stim,Apeak=Apeak,
				Ax=stimCenter2,AxWidth=AxWidth,
				Atheta=stimOrientation2,AthetaWidth=AthetaWidth,
				showActivityMaps=0,showModelParameters=0)
			attCRF[c] = R1[j,i]
			unattCRF[c] = R2[j,i]

		f = plt.figure(figsize=(7,7))
		s = f.add_subplot(121)
		plt.plot(contrasts,unattCRF)
		plt.plot(contrasts,attCRF)
		s.set_xscale('log')
		# plt.ylim([0, 7])
		plt.xlim(cRange)
		plt.legend(['Att non-pref','Att pref'],loc='best')
		plt.ylabel('Normalized response')
		plt.xlabel('Log contrast')
		plt.title(titleString)
		sn.despine(offset=10)
		s = f.add_subplot(122)
		plt.plot(contrasts,100*(attCRF-unattCRF)/unattCRF)
		s.set_xscale('log')
		plt.ylim([0, 100])
		plt.xlim(cRange)
		plt.ylabel('Attentional modulation (%)')
		plt.xlabel('Log contrast')
		sn.despine(offset=10)
		plt.tight_layout()


	def Figure5C(self):

		NMA = NormalizationModelofAttention()

		titleString = 'Figure 5C (McAdams and Maunsell, 1999)'
		stimWidth = 10
		AxWidth = 10

		# Sampling of space and orientation
		x =  np.mat(np.arange(-200,201))
		theta = np.mat(np.arange(-180,181)).T

		# Make stimuli
		stimCenter1 = 100
		stimOrientation1 = 0
		stimCenter2 = -100
		stimOrientation2 = 0
		stim1 = NMA.make_gaussian(theta,stimOrientation1,1,1) * NMA.make_gaussian(x,stimCenter1,stimWidth,1) 
		stim2 = NMA.make_gaussian(theta,stimOrientation2,1,1) * NMA.make_gaussian(x,stimCenter2,stimWidth,1)

		# Set contrast to 1 
		contrast = 1
		stim = contrast * stim1 * contrast + stim2

		# Population response when attending stim 1
		R1 = NMA.attention_model(x,theta,stim,Ax=stimCenter1,AxWidth=AxWidth)

		# Population response when attending stim 2
		R2 = NMA.attention_model(x,theta,stim,Ax=stimCenter2,AxWidth=AxWidth)

		# Pick RF center, record from neurons with that RF center and all
		# different feature preferences (same as tuning curve from any one of those
		# neurons).
		i = np.where((np.ravel(x)==stimCenter1))[0][0]
		attCRF = R1[:,i]
		unattCRF = R2[:,i]
		  
		f = plt.figure(figsize=(7,7))
		s = f.add_subplot(111)
		plt.plot(theta,unattCRF)
		plt.plot(theta,attCRF)
		plt.xlim([-180, 180])
		plt.legend(['Att Away','Att RF'])
		plt.title(titleString)
		sn.despine(offset=10)
		plt.tight_layout()


	def Figure6C(self):

		NMA = NormalizationModelofAttention()

		titleString = 'Figure 6C (Martinez-Trujillo & Treue, 2004)'
		stimWidth = 10 
		AxWidth = 30
		AthetaWidth = 60
		Ashape = 'cross'

		# Sampling of space and orientation
		x =  np.mat(np.arange(-200,201))
		theta = np.mat(np.arange(-180,181)).T

		# Make stimuli
		stimCenter1 = 100
		stimOrientation1 = 0
		stimCenter2 = -100
		stimOrientation2 = 0
		stim1 = NMA.make_gaussian(theta,stimOrientation1,1,1) * NMA.make_gaussian(x,stimCenter1,stimWidth,1) 
		stim2 = NMA.make_gaussian(theta,stimOrientation2,1,1) * NMA.make_gaussian(x,stimCenter2,stimWidth,1)

		# Set contrast to 1 
		contrast = 1  
		stim = contrast * stim1 * contrast + stim2

		# Population response when attending to fixation
		R1 = NMA.attention_model(x,theta,stim,Ax=0,AxWidth=AxWidth)

		# Population response when attending stim 2
		R2 = NMA.attention_model(x,theta,stim,Ashape='cross',
		  Ax=stimCenter2,AxWidth=AxWidth,
		  Atheta=stimOrientation2,AthetaWidth=AthetaWidth)

		# Pick RF center, record from neurons with that RF center and all
		# different feature preferences (same as tuning curve from any one of those
		# neurons).
		i = np.where((np.ravel(x)==stimCenter1))[0][0]
		attCRF = R1[:,i]
		unattCRF = R2[:,i]
		  
		f = plt.figure(figsize=(7,7))
		s = f.add_subplot(111)
		plt.plot(theta,unattCRF)
		plt.plot(theta,attCRF)
		plt.xlim([-180, 180])
		plt.legend(['Att Away','Att RF'])
		plt.title(titleString)
		sn.despine(offset=10)
		plt.tight_layout()



	def Figure7C(self):

		NMA = NormalizationModelofAttention()
		titleString = 'Figure 7C (Treue & Martinez-Trujillo, 1999)'
		stimWidth = 5 
		AxWidth = 5
		AthetaWidth = 45
		Apeak = 5

		# Sampling of space and orientation
		x =  np.mat(np.arange(-200,201))
		theta = np.mat(np.arange(-180,181)).T

		# Make stimuli 
		stimCenter1 = 93
		stimCenter2 = 107
		att_away_loc = -100
		RF_center = round(np.mean([stimCenter1,stimCenter2]))

		pair_att_vars  = []
		pair_att_nulls = []
		pair_att_aways = []
		Var_att_vars    = []
		Null_att_nulls  = []
		Var_att_aways   = []

		# Set contrast to 1
		contrast = 1  

		# Pick neuron to record
		j = np.where(np.ravel(theta)==0)[0][0]
		i = np.where(np.ravel(x)==RF_center)[0][0]

		orientations = np.linspace(-180,180,self.numOrientations)
		for stimOrientation1 in orientations:

		  stimOrientation2 = 180

		  stim1 = contrast * NMA.make_gaussian(theta,stimOrientation1,1,1) * NMA.make_gaussian(x,stimCenter1,stimWidth,1)
		  stim2 = contrast * NMA.make_gaussian(theta,stimOrientation2,1,1) * NMA.make_gaussian(x,stimCenter2,stimWidth,1)
		  pair = stim1 + stim2

		  # Population response when attending stim 1 (varying stim)
		  Pair_resp_att_var_pop = NMA.attention_model(x,theta,pair,Apeak=Apeak,
		    Ax=stimCenter1,AxWidth=AxWidth,
		    Atheta=stimOrientation1,AthetaWidth=AthetaWidth)

		  # Population response when attending stim 2 (null stim)
		  Pair_resp_att_null_pop = NMA.attention_model(x,theta,pair,Apeak=Apeak,
		    Ax=stimCenter2,AxWidth=AxWidth,
		    Atheta=stimOrientation2,AthetaWidth=AthetaWidth)

		  # Population response when attending to fixation point
		  Pair_resp_att_away_pop = NMA.attention_model(x,theta,pair,Apeak=Apeak,
		    Ax=att_away_loc,AxWidth=AxWidth,
		    Atheta=np.nan)

		  # Population response, attention to var presented alone
		  Var_att_var_pop = NMA.attention_model(x,theta,stim1,Apeak=Apeak,
		    Ax=stimCenter1,AxWidth=AxWidth,
		    Atheta=stimOrientation1,AthetaWidth=AthetaWidth)

		  # Population response, attention to null presented alone
		  Null_att_null_pop = NMA.attention_model(x,theta,stim2,Apeak=Apeak,
		    Ax=stimCenter2,AxWidth=AxWidth,
		    Atheta=stimOrientation2,AthetaWidth=AthetaWidth)

		  # Population response, attention away, var alone
		  Var_att_away_pop = NMA.attention_model(x,theta,stim1,Apeak=Apeak,
		    Ax=att_away_loc,AxWidth=AxWidth,
		    Atheta=np.nan)

		  pair_att_var    = Pair_resp_att_var_pop[j,i]
		  pair_att_null   = Pair_resp_att_null_pop[j,i]
		  pair_att_away   = Pair_resp_att_away_pop[j,i]
		  Var_att_var     = Var_att_var_pop[j,i]
		  Null_att_null   = Null_att_null_pop[j,i]
		  Var_att_away    = Var_att_away_pop[j,i]

		  pair_att_vars.append(pair_att_var)
		  pair_att_nulls.append(pair_att_null)
		  pair_att_aways.append(pair_att_away)
		  Var_att_vars.append(Var_att_var)
		  Null_att_nulls.append(Null_att_null)
		  Var_att_aways.append(Var_att_away)

		f = plt.figure(figsize=(7,7))
		plt.plot(orientations,pair_att_vars)
		plt.plot(orientations,pair_att_nulls)
		plt.plot(orientations,pair_att_aways)
		plt.plot(orientations,Var_att_vars)
		plt.plot(orientations,Null_att_nulls)
		plt.plot(orientations,Var_att_aways)
		plt.xlim([-180, 180])
		plt.legend(['Pair Var','Pair Null','Pair Away','Var var','Null Null','Var away'])
		plt.title(titleString)
		sn.despine(offset=10)
		plt.tight_layout()
