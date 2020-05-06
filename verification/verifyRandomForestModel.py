#!/usr/bin/python3

import sys
import time
import os
import datetime
import glob
import pandas as pd
import csv
import numpy as np
from sklearn import metrics
from sklearn.metrics import cohen_kappa_score as kappa
import sklearn.ensemble as ens
from sklearn.model_selection import train_test_split
#import matplotlib
#matplotlib.use('tkagg')
import matplotlib.pyplot as plt
from scipy import stats
from datetime import datetime
import pickle
from sklearn.metrics import brier_score_loss
from sklearn.calibration import calibration_curve

siteToName = {'KPGA': 'KPGA Representing Lake Powell Vicinity',
              'KBCE': 'KBCE Representing Southern Utah',
	      'K4HV': 'K4HV Representing Capitol Reef NP Vicinity',
	      'KSGU': 'KSGU Representing Zion NP Vicinity',
	      }

availableYears = [2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018]
trainingYearsList = [2011, 2012, 2013, 2014, 2015]		      			  
testingYearsList = [x for x in availableYears if x not in trainingYearsList]

sites = ["K4HV","KBCE","KPGA","KSGU"]

THERMO = ['MUCAPE', 'DCAPE', '7-5LR', 'LCL','PW', 'MeanRH','DD', 'WC', 'SfcTd', 'SfcT']

KIN = ['0-6km Shear Dir', '0-6km Shear Mag', 'C Dn Dir',
       'C Dn Mag', 'C Up Dir', 'C Up Mag', 'SM Dir','SM Mag',
       'MW Dir','MW Mag', '4-6km dir',
       '4-6km mag']

model = "NAM"

for chosenSite in sites:

   filename = "../Random Forest/savedModel_" + chosenSite + ".pck"
   clf = pickle.load(open(filename, 'rb'))

   plt.figure() 
   ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
   ax2 = plt.subplot2grid((3, 1), (2, 0))
   ax1.plot([0, 1], [0, 1], "k:", label="Perfectly Calibrated (Brier Score = 0)")

   # use *clean_nam for NAM forecasts, *full_clean for RAP forecasts
   if model == "NAM":
      #fileString = '/common/2018FFW_study/*clean_nam.csv'
      fileString = '/apps/2018FFW_study/csv/*.csv'
   elif model == "RAP":
      fileString = '/common/2018FFW_study/*full_clean.csv'
   for csv in glob.glob(fileString):

      probabilisticOutput = []
      deterministicOutput = []

      if model == "NAM":
	 #site = csv[-18:-14]
         site = csv[-8:-4]
      elif model == "RAP":
         site = csv[-19:-15]

      if site == '/4HV':
         site = 'K4HV'
      elif site == '4HV_':
         site = 'K4HV'

      if site != chosenSite:
         continue

      df = pd.read_csv(csv)

      if site == 'KBCE':
         df.loc[(df['Basins'] == 3), 'Result'] = 1
      if '4HV' or 'SGU' in site_name:
         df.loc[(df['Basins'] == 1), 'Result'] = 1
      test_df = df.dropna(subset=KIN + THERMO + ['Basins'] + ['Result'] + ['deterministicRRA'] + ['probabilisticRRA'])

      for year in trainingYearsList:
         test_df = test_df[test_df['Year'] != year] 	  
      xtest = test_df[THERMO + KIN]
      ytest = test_df.pop('Result')

      predictors = test_df[THERMO + KIN]

      yprob = clf.predict_proba(xtest)[0:, 1]
      yprob2 = test_df['probabilisticRRA']
      yprob3 = (test_df['probabilisticRRA'] + yprob)/2

      y_pred = clf.predict(xtest)
      #y_pred3 = round(yprob3)

      auc = metrics.roc_auc_score(ytest, yprob)
      brier_score = metrics.brier_score_loss(ytest, yprob)
      accuracy = metrics.accuracy_score(ytest, y_pred)
      cm = metrics.confusion_matrix(ytest, y_pred)   
      index_names = ['No Flooding Observed', 'Flooding Observed']
      col_names = ['No Flooding Forecast', 'Flooding Forecast']
      cm_df = pd.DataFrame(cm, columns =col_names, index=index_names)
      ck = kappa(ytest, y_pred)
      print(chosenSite)
      print('Probabilistic Verification')
      print('Brier Score {:3.2f}, '.format(brier_score) +
            'AUC = {:3.2f} '.format(auc))
      print('Deterministic Verification')
      print('Accuracy = {:3.2f}, '.format(accuracy) +
            'Cohen\'s Kappa {:3.2f}'.format(ck))
      print(cm_df.head())
      print('\n\n')

      clf_score = brier_score_loss(ytest, yprob)
      fraction_of_positives, mean_predicted_value = calibration_curve(ytest, yprob, n_bins=5)
      z = np.polyfit(mean_predicted_value, fraction_of_positives, 1)
      f = np.poly1d(z)
      x = np.array([0.125, 0.375, 0.605, 0.845])
      y = f(x)
      ax1.plot(x, y, "s-", label="%s (Brier = %1.3f)" % ("Random Forest Model", clf_score))
      #ax2.hist(yprob, range=(0, 1), bins=10, histtype="step", lw=2)

      clf_score = brier_score_loss(ytest, yprob2)
      fraction_of_positives, mean_predicted_value = calibration_curve(ytest, yprob2, n_bins=5)
      z = np.polyfit(mean_predicted_value, fraction_of_positives, 1)
      f = np.poly1d(z)
      x = np.array([0.125, 0.375, 0.605, 0.845])
      y = f(x)
      ax1.plot(x, y, "s-", label="%s (Brier = %1.3f)" % ("Legacy RRA", clf_score))
      #ax2.hist(yprob2, range=(0, 1), bins=10, histtype="step", lw=2)

      clf_score = brier_score_loss(ytest, yprob3)
      fraction_of_positives, mean_predicted_value = calibration_curve(ytest, yprob3, n_bins=5)
      z = np.polyfit(mean_predicted_value, fraction_of_positives, 1)
      f = np.poly1d(z)
      x = np.array([0.125, 0.375, 0.605, 0.845])
      y = f(x)
      ax1.plot(x, y, "s-", label="%s (Brier = %1.3f)" % ("Blended RRA & Random Forest Model", clf_score))
      #ax2.hist(yprob3, range=(0, 1), bins=10, histtype="step", lw=2)
      
      ax2.text(0.1, 0.1, cm_df.head())
      ax2.axis('off')
      
      bbox = dict(boxstyle="round", fc="0.8")
      ax1.annotate('Not Expected', (0.125-0.055, -0.05), bbox=bbox)
      bbox = dict(boxstyle="round", fc="0.8")
      ax1.annotate('Possible', (0.375-0.04, -0.05), bbox=bbox)
      bbox = dict(boxstyle="round", fc="0.8")
      ax1.annotate('Probable', (0.605-0.04, -0.05), bbox=bbox)
      bbox = dict(boxstyle="round", fc="0.8")
      ax1.annotate('Expected', (0.845-0.04, -0.05), bbox=bbox)

   ax1.set_title(siteToName[chosenSite])
   ax1.set_xlabel('Probabilistic Forecast')
   ax1.set_ylabel('Observed Percentage')
   ax1.legend()   
   
   del(df)
   del(test_df)   
   
plt.show()
