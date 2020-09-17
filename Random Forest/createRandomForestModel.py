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
from sklearn.model_selection import RandomizedSearchCV

availableYears = [2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020]

print('\n')
response = input("Select a method to divide the training data. \n" + \
                 "    Enter '1' for random, '2' or just <enter> for specific years: ")
print('\n')

if len(response) == 0 or response == '2':
   response2 = input("Enter years for training, either comma separated for each year or a range divided by a dash. \n" +
               "    Default is '2011-2016' for training, or just <enter>: ")
   print('\n')
   if ',' in response2:
      trainingYearsList = list(map(int, response2.split(",")))
   elif '-' in response2:
      firstYear = datetime.strptime(str(response2[0:response2.find('-')]),'%Y')
      secondYear = datetime.strptime(str(response2[response2.find('-')+1:len(response2)]),'%Y')
      trainingYearsList = range(firstYear.year, secondYear.year+1)
   elif len(response2) == 0:
      trainingYearsList = [2011, 2012, 2013, 2014, 2015, 2016]
   else:
      print("please use the format indicated in the prompts")
      sys.exit()   
   testingYearsList = [x for x in availableYears if x not in trainingYearsList]

response3 = input("Run on each subdivided group of basins (e.g. KSGU) or all at once (KBCE only)?\n" + \
                  "   If running on all at once, you will have the option to run the program iteratively\n" + \
		  "   to obtain a desired skill level.\n" + \
                  "   Enter '1' to see the results for each basin, '2' or just <enter> for all in one:\n ") 
print('\n')

response4 = input("Select a model for the BUFR sounding. \n" + \
                  "   Enter '1' for RAP, '2' or just <enter> for NAM: ")
print('\n')

#response5 = input("Display plots of # of basins flooded vs probabilistic output? \n" + \
#                  "   Enter '1' for yes, '2' or just <enter> for no: ")
#print('\n')

response5 = input("Display reliability plot of # of flooding vs probabilistic output? \n" + \
                  "   Enter '1' for yes, '2' or just <enter> for no: ")
print('\n')

response6 = input("Calculate stats for RRA and blended RRA/RandomForest? \n" + \
                  "   Enter '1' for no, '2' or just <enter> for yes: ")
print('\n')

response7 = input("Run multiple times (iteratively) to reach a target accuracy? \n" + \
                  "   Enter '1' for yes, '2' or just <enter> for no: ")
print('\n')

if response7 == '1':
    response8 = input("Target metric for an 'accurate' forecast: \n" + \
                      "   Enter '1' for Cohen's Kappa (deterministic), '2' or just <enter> for Brier (probabilistic): ")
    print('\n')
    
    if response8 == '1':
        response9 = input("Target Cohen's Kappa: \n" + \
                          "   E.g, 0.35: ")
        print('\n')
    else:
        response9 = input("Target Brier Score: \n" + \
                          "   E.g, 0.07: ")
        print('\n') 		      			  

response10 = input("Save the Random Forest Model for the RRA tool? \n" + \
                  "   Enter '1' for yes, '2' or just <enter> for no: ")
print('\n') 

response11 = input("Search for optimal hyperparameters (increases runtime)? \n" + \
                  "   Enter '1' for yes, '2' or just <enter> for no: ")
print('\n')

response12 = input("Use XGBoost in place of standard random forest? \n" + \
                  "   Enter '1' for yes, '2' or just <enter> for no: ")
print('\n')
 

sites = ["K4HV","KBCE","KPGA","KSGU"]

# percent of dataset to turn into test cases for wetter and drier locations based on number of flash floods
drySitesPercentage = 30
wetSitesPercentage = 30

drySites = ["K4HV","KSGU"]

# default to the wetter sites percentage
percentToTest = wetSitesPercentage

THERMO = ['MUCAPE', 'DCAPE', '7-5LR', 'LCL','PW', 'MeanRH','DD', 'WC', 'SfcTd', 'SfcT']

KIN = ['0-6km Shear Dir', '0-6km Shear Mag', 'C Dn Dir',
       'C Dn Mag', 'C Up Dir', 'C Up Mag', 'SM Dir','SM Mag',
       'MW Dir','MW Mag', '4-6km dir',
       '4-6km mag']

def createReliabilityPlots(site, response5, response6, ytest, yprob, yprob2, yprob3):
      if response5 == '2' or len(response5) == 0:
         clf_score = brier_score_loss(ytest, yprob)
         fraction_of_positives, mean_predicted_value = calibration_curve(ytest, yprob, n_bins=10)
         fig, axs = plt.subplots(2)
         plt.suptitle('Reliability Plot for {}'.format(site), fontweight='bold')
         axs[0].plot([0, 1], [0, 1], "k:", label="Perfectly Calibrated")
         axs[0].plot(mean_predicted_value, fraction_of_positives, "s-", label="%s (%1.3f)" % ("Random Forest Model for " + site, clf_score))
         axs[0].set_ylabel('Fraction of Positives', fontweight='bold') 
         axs[0].set_xlabel('Mean Predicted Value', fontweight='bold') 
         axs[0].legend(loc='upper left')
         axs[1].hist(yprob, range=(0, 1), bins=10, label=site,
             histtype="step", lw=2)
         axs[1].set_xlabel("Mean predicted value", fontweight='bold')
         axs[1].set_ylabel("Count", fontweight='bold')
         axs[1].set_yscale('log')
         # Add plots for legacy RRA product comparison, if requested
         if response6 == '2' or len(response6) == 0:
            clf_score = brier_score_loss(ytest, yprob2)
            fig, axs = plt.subplots(2)
            fraction_of_positives, mean_predicted_value = calibration_curve(ytest, yprob2, n_bins=10)
            axs[0].plot([0, 1], [0, 1], "k:", label="Perfectly Calibrated")
            axs[0].plot(mean_predicted_value, fraction_of_positives, "s-", label="%s (%1.3f)" % ("Legacy RRA", clf_score))
            axs[0].set_ylabel('Fraction of Positives', fontweight='bold') 
            axs[0].set_xlabel('Mean Predicted Value', fontweight='bold') 
            axs[0].legend(loc='upper left')
            axs[1].hist(yprob2, range=(0, 1), bins=10, label=site,
            histtype="step", lw=2)
            axs[1].set_xlabel("Mean predicted value", fontweight='bold')
            axs[1].set_ylabel("Count", fontweight='bold')
            axs[1].set_yscale('log')

            clf_score = brier_score_loss(ytest, yprob3)
            fig, axs = plt.subplots(2) 
            fraction_of_positives, mean_predicted_value = calibration_curve(ytest, yprob3, n_bins=10)
            axs[0].plot([0, 1], [0, 1], "k:", label="Perfectly Calibrated")
            axs[0].plot(mean_predicted_value, fraction_of_positives, "s-", label="%s (%1.3f)" % ("Blended RRA & Random Forest", clf_score))
            axs[0].set_ylabel('Fraction of Positives', fontweight='bold') 
            axs[0].set_xlabel('Mean Predicted Value', fontweight='bold') 
            axs[0].legend(loc='upper left')
            axs[1].hist(yprob3, range=(0, 1), bins=10, label=site,
            histtype="step", lw=2)
            axs[1].set_xlabel("Mean predicted value", fontweight='bold')
            axs[1].set_ylabel("Count", fontweight='bold')
            axs[1].set_yscale('log')  


def tuneHyperparameters(xtrain, ytrain):
    n_estimators = [int(x) for x in np.linspace(start = 200, stop = 20000, num = 10)]
    max_features = ['auto', 'sqrt']
    max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
    min_samples_split = [2, 5, 10]
    min_samples_leaf = [1, 2, 4]
    bootstrap = [True, False]
    
    random_grid = {'n_estimators': n_estimators,
                   'max_features': max_features,
		   'max_depth': max_depth,
		   'min_samples_split': min_samples_split,
		   'min_samples_leaf': min_samples_leaf,
		   'bootstrap': bootstrap}
    # Option to use a 'tradition' random forest, or boosted gradient trees via XGBoost
    if len(response12) == 0 or response12 == '2':
        clf = ens.RandomForestClassifier()
        max_depth.append(None) # Per error message, this seems to only be applicable to the RandomForestClassifier
    elif response12 == '1':
        import xgboost as xgb
        clf = xgb.XGBClassifier()
    clfRandom = RandomizedSearchCV(estimator = clf, param_distributions = random_grid, n_iter = 100,
                                   cv = 3, random_state = 42, n_jobs = -1)
    clfRandom.fit(xtrain, ytrain)
    print(clfRandom.best_params_)
    return(clfRandom)		   

if len(response4) == 0 or response4 == '2':
   model = "NAM"
else:
   model = "RAP"

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

   if response3 == '2' or len(response3) == 0:
      if site != 'KBCE':
         continue

   # loop as needed to reach target metrics
   targetAccuracy = False   
   while not targetAccuracy:

      if site in drySites:
         percentToTest = drySitesPercentage
      else:
         percentToTest = wetSitesPercentage

      df = pd.read_csv(csv)
      
      # Trim to Nick's RAP dates
      #dfTrimmed = pd.read_csv(csv.replace('clean_nam', 'full_clean'))   
      #dfIntersection = pd.merge(df, dfTrimmed, how='inner', on=['Date'], suffixes=('','_y'))
      #cols = [c for c in dfIntersection.columns if c.lower()[-2:] != '_y']
      #dfIntersection = dfIntersection[cols]
      #dfInsersection = dfIntersection[dfIntersection.columns.drop('Unnamed: 0')]
      #df = dfIntersection
      
      if site == 'KBCE':
         df.loc[(df['Basins'] == 0), 'Result'] = 0
         df.loc[(df['Basins'] >= 1), 'Result'] = 1
      elif site == "K4HV" or site == "KSGU":
         df.loc[(df['Basins'] == 0), 'Result'] = 0
         df.loc[(df['Basins'] >= 1), 'Result'] = 1
      elif site == "KPGA":
         df.loc[(df['Basins'] == 0), 'Result'] = 0
         df.loc[(df['Basins'] >= 2), 'Result'] = 1

      # Filter out May/June/October
      df.loc[(df['Month'] == 5), 'Result'] = np.nan
      df.loc[(df['Month'] == 6), 'Result'] = np.nan
      df.loc[(df['Month'] == 10), 'Result'] = np.nan

      df2 = df.dropna(subset=KIN + THERMO + ['Basins'] + ['Result'] + ['deterministicRRA'] + ['probabilisticRRA'])

      result = df2['Result']
      predictors = df2[THERMO + KIN]

      if response == '1': # use random split
         xtrain, xtest, ytrain, ytest = train_test_split(predictors, result, test_size = percentToTest/100)
         test_df = df.iloc[xtest.index.tolist()]

      if len(response) == 0 or response == '2':
         training_df = df2
         test_df = df2
         for year in testingYearsList:
            training_df = training_df[training_df['Year'] != year]
         for year in trainingYearsList:
            test_df = test_df[test_df['Year'] != year] 	  
         xtrain = training_df[THERMO + KIN]
         xtest = test_df[THERMO + KIN]
         ytrain = training_df.pop('Result')
         ytest = test_df.pop('Result')

      if len(response11) == 0 or response11 == '2':
         if len(response12) == 0 or response12 == '2':
             # The hyperparameters below are from running hyperparameter optimization for KBCE
             clf = ens.RandomForestClassifier(n_estimators=2400, max_depth=30, min_samples_leaf=1,
                                              min_samples_split=5,oob_score=True, class_weight='balanced')
         elif response12 == '1':
             import xgboost as xgb
             # The hyperparameters below are from running hyperparameter optimization for KBCE
             clf = xgb.XGBClassifier(n_estimators=20000, min_samples_split=10, min_samples_leaf=2,
                                     max_features='sqrt', max_depth=10, bootstrap=True)
         clf.fit(xtrain, ytrain)					
      else:
         clf = tuneHyperparameters(xtrain, ytrain)			     

      y_pred = clf.predict(xtest)
      accuracy = metrics.accuracy_score(ytest, y_pred)
      yprob = clf.predict_proba(xtest)[0:, 1]
      #yprob = (yprob+yprob2)/2
      brier_score = metrics.brier_score_loss(ytest, yprob)
      auc = metrics.roc_auc_score(ytest, yprob)
      cm = metrics.confusion_matrix(ytest, y_pred)
      index_names = ['No Flooding Observed', 'Flooding Observed']
      col_names = ['No Flooding Forecast', 'Flooding Forecast']
      cm_df = pd.DataFrame(cm, columns =col_names, index=index_names)
      ck = kappa(ytest, y_pred)
      print('\n\n')
      if 'trainingYearsList' in locals():
         print('FFW model results for ' + site + ' using the ' + model + ', testing on ' + str(testingYearsList))
      else:
         print('FFW model results for ' + site + ' using the ' + model + ', testing on ' + str(percentToTest) + '% of the 2011-2018 dataset')
      print('--Probabilistic Verification--')
      print('Brier Score {:3.2f}, '.format(brier_score) +
             'AUC = {:3.2f} '.format(auc))
      print('--Deterministic Verification--')
      print('Accuracy = {:3.2f}, '.format(accuracy) +
              'Cohen\'s Kappa {:3.2f}'.format(ck))
      print(cm_df.head())

      if response6 == '2' or len(response6) == 0:
         yprob2 = test_df['probabilisticRRA']
         y_pred2 = test_df['deterministicRRA']
         auc = metrics.roc_auc_score(ytest, yprob2)
         brier_score = metrics.brier_score_loss(ytest, yprob2)
         accuracy = metrics.accuracy_score(ytest, y_pred2)
         cm = metrics.confusion_matrix(ytest, y_pred2)   
         cm_df = pd.DataFrame(cm, columns =col_names, index=index_names)
         ck = kappa(ytest, y_pred2)
         if 'trainingYearsList' in locals():
            print('\nSLC RRA verification for ' + site + ' using the ' + model + ', testing on ' + str(testingYearsList))
         else:
            print('\nSLC RRA verification for ' + site + ' using the ' + model + ', testing on ' + str(percentToTest) + '% of the 2011-2018 dataset')
         print('--Probabilistic Verification--')
         print('Brier Score {:3.2f}, '.format(brier_score) +
                'AUC = {:3.2f} '.format(auc))
         print('--Deterministic Verification--')
         print('Accuracy = {:3.2f}, '.format(accuracy) +
                 'Cohen\'s Kappa {:3.2f}'.format(ck))
         print(cm_df.head())

         yprob3 = (test_df['probabilisticRRA'] + yprob)/2
         y_pred3 = round(yprob3)
         auc = metrics.roc_auc_score(ytest, yprob3)
         brier_score = metrics.brier_score_loss(ytest, yprob3)
         accuracy = metrics.accuracy_score(ytest, y_pred3)
         cm = metrics.confusion_matrix(ytest, y_pred3)   
         cm_df = pd.DataFrame(cm, columns =col_names, index=index_names)
         ck = kappa(ytest, y_pred3)
         if 'trainingYearsList' in locals():   
            print('\nSLC combined verification for ' + site + ' using the ' + model + ', testing on ' + str(testingYearsList))
         else:
            print('\nSLC combined verification for ' + site + ', testing on ' + str(percentToTest) + '% of the 2011-2018 dataset')
         print('--Probabilistic Verification--')
         print('Brier Score {:3.2f}, '.format(brier_score) +
        	'AUC = {:3.2f} '.format(auc))
         print('--Deterministic Verification--')
         print('Accuracy = {:3.2f}, '.format(accuracy) +
        	 'Cohen\'s Kappa {:3.2f}'.format(ck))
         print(cm_df.head())

      print('\n\n')

      saveFile = 'savedModel_' + site + '.pck'
      if response7 != '1':
         targetAccuracy = True
         createReliabilityPlots(site, response5, response6, ytest, yprob, yprob2, yprob3)
         if response10 == '1':
            pickle.dump(clf, open(saveFile, 'wb'))	 	 
      else:
         if response8 == '1':
            if ck > float(response9):
               targetAccuracy = True
               createReliabilityPlots(site, response5, response6, ytest, yprob, yprob2, yprob3)
               if response10 == '1':
                   pickle.dump(clf, open(saveFile, 'wb'))
            else:
               print("Didn't reach target Cohen's Kappa of "+str(response9)+", only reached "+str(ck)+", trying again")	
         else:
            if brier_score < float(response9):
               targetAccuracy = True
               createReliabilityPlots(site, response5, response6, ytest, yprob, yprob2, yprob3)
               if response10 == '1':
                   pickle.dump(clf, open(saveFile, 'wb'))	       
            else:
               print("Didn't reach target Brier Skill Score of "+str(response9)+", only reached "+str(brier_score)+", trying again")	       		 


if response5 == '1':
   plt.show()
