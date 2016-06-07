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


class NormalizationModelofAttention(object):

	# def __init__(self,):

	# 	self.

	def make_gaussian(self, space, center, width, heigth=np.nan):
		if not np.isnan(heigth):
			kernel = heigth*np.exp(-np.power(np.array(space) - center, 2.) / (2 * np.power(width, 2.)))
		else:
			kernel = np.exp(-np.power(np.array(space) - center, 2.) / (2 * np.power(width, 2.)))
			kernel /= np.sum(kernel)
		return kernel

	def conv_2_sep_Y_circ(self,image,xkernel,thetakernel):#,upsample=10):
		# up_image = np.repeat(image,upsample)
		x_convolved = ndimage.convolve1d(image,xkernel,mode='constant',axis=1)
		both_convolved = ndimage.convolve1d(x_convolved,thetakernel,mode='wrap',axis=0)
		return both_convolved

	def attention_model(self,x,theta,stimulus,ExWidth = 5,EthetaWidth = 60,IxWidth = 20,
		IthetaWidth = 360,Ax = np.nan,Atheta = np.nan,AxWidth = np.nan,AthetaWidth = np.nan,
		Apeak =  2,Abase = 1,Ashape = 'oval',sigma = 1e-6,baselineMod = 0,baselineUnmod = 0,
		showActivityMaps = 0,showModelParameters = 0):
		"""
		R = attentionModel(x,theta,stimulus,[param1],[value1],[param2],[value2],...,[paramN],[valueN])

		Required arguments
		x is a row vector of spatial coordinates
		theta is a column vector of feature/orientation coordinates
		stimulus is NxM where N is the length of theta and M is the length of x

		Optional parameters are passed as string/value pairs. If any of them are
		not specified then default values are used. Valid parameters are as
		follows.

		ExWidth: spatial spread of stimulation field
		EthetaWidth: feature/orientation tuning width of stimulation field
		IxWidth: spatial spread of suppressive field
		IthetaWidth: feature/orientation extent/width of suppressive field
		Ax: spatial center of attention field
		Atheta: feature/orientation center of attention field
		AxWidth: spatial extent/width  attention field
		AthetaWidth: feature/orientation extent/width  attention field
		Apeak: peak amplitude of attention field
		Abase: baseline of attention field for unattended locations/features
		Ashape: either 'oval' or 'cross'
		sigma: constant that determines the semi-saturation contrast
		baselineMod: amount of baseline added to stimulus drive
		baselineUnmod: amount of baseline added after normalization
		showActivityMaps: if non-zero, then display activity maps
		showModelParameters: if non-zero, then display stimulus, stimulation
		field, suppressive field, and attention field.

		If Ax or Atheta are NaN or not specified then attention is spread evenly
		such that attnGain = 1 (a constant) for all spatial positions or
		features/orientations, respectively.

		Returns the population response (R), same size as stimulus, for neurons
		with receptive fields centered at each spatial position and tuned to each
		feature/orientation.
		"""

		if np.isnan(AxWidth):
			Axwidth = ExWidth
		if np.isnan(AthetaWidth):
			AthetaWidth = EthetaWidth

		# Stimulation field and suppressive field
		ExKernel = self.make_gaussian(x,0,ExWidth)
		IxKernel = self.make_gaussian(x,0,IxWidth)
		EthetaKernel = self.make_gaussian(theta,0,EthetaWidth)
		IthetaKernel = self.make_gaussian(theta,0,IthetaWidth)

		# Attention field
		if np.isnan(Ax) * np.isnan(Atheta):
			attnGain = np.ones(size(stimulus))
		else:
			if np.isnan(Ax):
				attnGainX = np.ones(len(np.ravel(x)))
			else:
				attnGainX = self.make_gaussian(x,Ax,AxWidth,1)
				if Ashape == 'cross':
					attnGainX = (Apeak-Abase)*attnGainX + Abase
			if np.isnan(Atheta):
				attnGainTheta = np.ones(len(theta))
				Atheta = 0
			else:
				attnGainTheta = self.make_gaussian(theta,0,AthetaWidth,1)
				if Ashape == 'cross':
					attnGainTheta = (Apeak-Abase)*attnGainTheta + Abase

			impulse = (theta == Atheta)
			tmp = impulse * attnGainX
			attnGain = self.conv_2_sep_Y_circ(tmp,[1],np.ravel(attnGainTheta))
			attnGain = (Apeak-Abase)*attnGain + Abase

		# Stimuulus drive
		Eraw = self.conv_2_sep_Y_circ(stimulus,np.ravel(ExKernel),np.ravel(EthetaKernel)) + baselineMod
		Emax = np.max(Eraw)
		E = attnGain * Eraw

		# Suppressive drive
		I = self.conv_2_sep_Y_circ(E,np.ravel(IxKernel),np.ravel(IthetaKernel))
		Imax = np.max(I)

		# Normalization
		R = E / (I + sigma) + baselineUnmod
		Rmax = np.max(R)

		if showModelParameters == 1:
			f=plt.figure(figsize=(9,9))
			s=f.add_subplot(221)
			plt.plot(np.ravel(x),np.ravel(ExKernel),label='ExKernel')
			plt.plot(np.ravel(x),-np.ravel(IxKernel),label='IxKernel')
			plt.legend(loc='best')
			sn.despine(offset=10)
			s=f.add_subplot(222)
			plt.plot(np.ravel(theta),np.ravel(EthetaKernel),label='EthetaKernel')
			plt.plot(np.ravel(theta),-np.ravel(IthetaKernel),label='IthetaKernel')			
			sn.despine(offset=10)
			plt.legend(loc='best')
			s=f.add_subplot(223)
			plt.plot(np.ravel(x),np.ravel(attnGainX),label='attnGainX')
			plt.legend(loc='best')
			sn.despine(offset=10)
			s=f.add_subplot(224)
			plt.plot(np.ravel(theta),np.ravel(attnGainTheta),label='attnGainTheta')		  
			plt.legend(loc='best')
			sn.despine(offset=10)
			plt.tight_layout()

		if showActivityMaps == 1:
			f = plt.figure(figsize=(9,12))
			s = f.add_subplot(321)
			plt.imshow(stimulus,interpolation='nearest',cmap='gray',clim=[0,1])
			plt.xlabel('Space')
			plt.ylabel('Orientation')
			plt.title('Stimulus')
			plt.xticks([])
			plt.yticks([])
			s = f.add_subplot(322)
			plt.imshow(Eraw,interpolation='nearest',cmap='gray',clim=[0,Emax])
			plt.xlabel('Receptive field center')
			plt.ylabel('Orientation preference')
			plt.title('Stimulus drive')			
			plt.xticks([])
			plt.yticks([])
			s = f.add_subplot(3,2,3)
			plt.imshow(attnGain,interpolation='nearest',cmap='gray',clim=[0,np.max(attnGain)])
			plt.xlabel('Receptive field center')
			plt.ylabel('Orientation preference')
			plt.title('Attention field')
			plt.xticks([])
			plt.yticks([])
			s = f.add_subplot(3,2,4)
			plt.imshow(I,interpolation='nearest',cmap='gray',clim=[0,Imax])
			plt.xlabel('Receptive field center')
			plt.ylabel('Orientation preference')
			plt.title('Suppressive drive')
			plt.xticks([])
			plt.yticks([])
			s = f.add_subplot(3,2,5)
			plt.imshow(R,interpolation='nearest',cmap='gray',clim=[0,Rmax])
			plt.xlabel('Receptive field center')
			plt.ylabel('Orientation preference')
			plt.title('Population response')
			plt.xticks([])
			plt.yticks([])
			plt.tight_layout()

		if (showActivityMaps + showModelParameters) > 0:
			plt.show()

		return R


