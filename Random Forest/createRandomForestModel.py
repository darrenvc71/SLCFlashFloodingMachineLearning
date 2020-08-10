#!/usr/bin/python3

import sys, os
import glob
import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.metrics import cohen_kappa_score as kappa
import sklearn.ensemble as ens
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from scipy import stats
from datetime import datetime
import pickle
from sklearn.metrics import brier_score_loss
from sklearn.calibration import calibration_curve
from sklearn.model_selection import RandomizedSearchCV

availableYears = [2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018]

sites = ["K4HV","KBCE","KPGA","KSGU"]

THERMO = ['MUCAPE', 'DCAPE', '7-5LR', 'LCL','PW', 'MeanRH','DD', 'WC', 'SfcTd', 'SfcT']

KIN = ['0-6km Shear Dir', '0-6km Shear Mag', 'C Dn Dir',
       'C Dn Mag', 'C Up Dir', 'C Up Mag', 'SM Dir','SM Mag',
       'MW Dir','MW Mag', '4-6km dir',
       '4-6km mag']

# If a random split is used, different percentages for the testing portion can be used
# Currently, both 'drier' and 'wetter' sites are set to use the same % for testing
drySites = ["K4HV","KSGU"]
drySitesPercentage = 30
wetSitesPercentage = 30
# Default to the wetter sites percentage
percentToTest = wetSitesPercentage

print('\n')
response = input("Select a method to divide the training data. \n" + \
                 "    Enter '1' for random, '2' or just <enter> for specific years: ")
print('\n')

if len(response) == 0 or response == '2':
   response2 = input("Enter years for training, either comma separated for each year or a range divided by a dash. \n" +
               "    Default is '2011-2015' for training, or just <enter>: ")
   print('\n')
   if ',' in response2:
      trainingYearsList = list(map(int, response2.split(",")))
   elif '-' in response2:
      firstYear = datetime.strptime(str(response2[0:response2.find('-')]),'%Y')
      secondYear = datetime.strptime(str(response2[response2.find('-')+1:len(response2)]),'%Y')
      trainingYearsList = range(firstYear.year, secondYear.year+1)
   elif len(response2) == 0:
      trainingYearsList = [2011, 2012, 2013, 2014, 2015]
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
response5 = input("Display reliability plot of # of flooding vs probabilistic output? \n" + \
                  "   Enter '1' for no, '2' or just <enter> for yes: ")
print('\n')
response6 = input("Calculate stats for RRA and blended RRA/RandomForest? \n" + \
                  "   Enter '1' for no, '2' or just <enter> for yes: ")
print('\n')
response7 = input("Search for optimal hyperparameters (increases runtime)? \n" + \
                  "   Note: 1 season will be subtracted from the training dataset for this\n" + \
                  "   Enter '1' for yes, '2' or just <enter> for no: ")
print('\n') 
response8 = input("Save the Random Forest Model for the RRA tool? \n" + \
                  "   Enter '1' for yes, '2' or just <enter> for no: ")
print('\n') 

def createReliabilityPlots(response5, response6, ytest, yprob, yprob2, yprob3):
      if response5 == '2' or len(response5) == 0:
         clf_score = brier_score_loss(ytest, yprob)
         fraction_of_positives, mean_predicted_value = calibration_curve(ytest, yprob, n_bins=10)
         plt.plot([0, 1], [0, 1], "k:", label="Perfectly Calibrated")
         plt.plot(mean_predicted_value, fraction_of_positives, "s-", label="%s (%1.3f)" % ("Random Forest Model", clf_score))
         plt.legend(loc='best')
         # Add plots for legacy RRA product comparison, if requested
         if response6 == '2' or len(response6) == 0:
            clf_score = brier_score_loss(ytest, yprob2)
            plt.figure() 
            fraction_of_positives, mean_predicted_value = calibration_curve(ytest, yprob2, n_bins=10)
            plt.plot([0, 1], [0, 1], "k:", label="Perfectly Calibrated")
            plt.plot(mean_predicted_value, fraction_of_positives, "s-", label="%s (%1.3f)" % ("Legacy RRA", clf_score))
            plt.legend(loc='best')

            clf_score = brier_score_loss(ytest, yprob3)
            plt.figure() 
            fraction_of_positives, mean_predicted_value = calibration_curve(ytest, yprob3, n_bins=10)
            plt.plot([0, 1], [0, 1], "k:", label="Perfectly Calibrated")
            plt.plot(mean_predicted_value, fraction_of_positives, "s-", label="%s (%1.3f)" % ("Blended RRA & Random Forest", clf_score))
            plt.legend(loc='best')

def tuneHyperparameters(xtrain, ytrain):
    n_estimators = [int(x) for x in np.linspace(start = 200, stop = 20000, num = 10)]
    max_features = ['auto', 'sqrt']
    max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
    max_depth.append(None)
    min_samples_split = [2, 5, 10]
    min_samples_leaf = [1, 2, 4]
    bootstrap = [True, False]
    
    random_grid = {'n_estimators': n_estimators, 'max_features': max_features, 'max_depth': max_depth,
                   'min_samples_split': min_samples_split, 'min_samples_leaf': min_samples_leaf,
                   'bootstrap': bootstrap}
    clf = ens.RandomForestClassifier()
    clfRandom = RandomizedSearchCV(estimator = clf, param_distributions = random_grid, n_iter = 100,
                                   cv = 3, random_state = 42, n_jobs = -1)
    clfRandom.fit(xtrain, ytrain)
    print(clfRandom.best_params_)
    return(clfRandom)	   

if len(response4) == 0 or response4 == '2':
   model = "NAM"
else:
   model = "RAP"

# Allow for the path from either the top level directory, or the Random Forest folder
if os.path.isdir('csv'):
    filePathPreprend = 'csv'
else:
    filePathPreprend = '../csv'

# Open the selected model's csv location
if model == "NAM":
   fileString = filePathPreprend + '/*.csv'
elif model == "RAP":
   fileString = filePathPreprend + '/*full_clean.csv'

for csv in glob.glob(fileString):
   
    probabilisticOutput = []
    deterministicOutput = []

    # Find the site name and fix if necessary
    if model == "NAM":
       site = csv[-8:-4]
    elif model == "RAP":
       site = csv[-19:-15]
    if site == '/4HV':
       site = 'K4HV'
    elif site == '4HV_':
       site = 'K4HV'

    # Only use the KBCE site if that's what was requested     
    if response3 == '2' or len(response3) == 0:
       if site != 'KBCE':
          continue

    if site in drySites:
       percentToTest = drySitesPercentage
    else:
       percentToTest = wetSitesPercentage

    df = pd.read_csv(csv)
      
    # Build a Results column if it doesn't exist already
    if 'Result' not in df.columns:
        df['Result'] = np.nan
    
    # Assign a value for the Results column (the predictand) based on the number of basins flooding
    # Still deciding how to handle this as there are not many days of >1 basin flooding when compared to non-flood days
    df.loc[(df['Basins'] == 0), 'Result'] = 0 
    # The code below works correctly, but is very limiting to the number of cases for training
    #if site == 'KBCE':
    #    df.loc[(df['Basins'] == 0), 'Result'] = 0
    #    df.loc[(df['Basins'] >= 1), 'Result'] = np.nan       
    #    df.loc[(df['Basins'] >= 3), 'Result'] = 1
    #elif site == "K4HV" or site == "KSGU":   
    #    df.loc[(df['Basins'] >= 1), 'Result'] = 1      
    #elif site == "KPGA": 
    #    df.loc[(df['Basins'] >= 2), 'Result'] = 1
    # Because the code above is quite limiting, using this alternative for now
    df.loc[(df['Basins'] >= 1), 'Result'] = 1 
    
    # Randomly remove many of the non-flash flooding dates
    #indexToKeep = list(df.index[df['Result'] == 1])
    #numberOfCases = int(df['Result'].sum())
    #dropIndicies = []
    #for x in range(1,len(df)-2*numberOfCases):
    #    dropIndicies.append(np.random.choice(np.setdiff1d(range(1,len(df)), indexToKeep.append(x))))
    #df = df.drop(dropIndicies)
            
    # Make a new dataframe that removes any dates with missing entries, set up the predictor & predictands   
    df2 = df.dropna(subset=KIN + THERMO + ['Basins'] + ['Result'] + ['deterministicRRA'] + ['probabilisticRRA'])  
    result = df2['Result']
    predictors = df2[THERMO + KIN]

    # Random split of dates option, otherwise use fixed dates
    if response == '1': 
        xtrain, xtest, ytrain, ytest = train_test_split(predictors, result, test_size = percentToTest/100)
        test_df = df.iloc[xtest.index.tolist()]
    elif len(response) == 0 or response == '2':
        # Slightly convoluted way to use the years the user specifies
        training_df = df2
        test_df = df2
        for year in testingYearsList:
            training_df = training_df[training_df['Year'] != year]
        for year in trainingYearsList:
            test_df = test_df[test_df['Year'] != year]
        xtrain = training_df[THERMO + KIN]
        xtest = test_df[THERMO + KIN]
        ytrain = training_df['Result']
        ytest = test_df['Result']

    # Print out a list of how many cases are being used in training/testing
    print('Using ' + str(int(ytrain.sum())) + ' FF cases for ' + site + ' training and ' + str(int(ytest.sum())) + ' cases for testing')

    # Create a random forest model using a default set of hyperparameters. If the user specifies, 
    # tune the hyperparameters first
    if len(response7) == 0 or response7 == '2':
        clf = ens.RandomForestClassifier(n_estimators=2400, max_depth=30, min_samples_leaf=1,
                                         min_samples_split=5,oob_score=True,class_weight='balanced')
        clf.fit(xtrain, ytrain)
    else:
        # We will need to remove a year to use for validation, assuming test/training are split by years
        # Simply removing last year of training for now
        if len(response) == 0 or response == '2':
            validationYear = trainingYearsList[-1]
            validation_df = df2
            validation_df = validation_df[validation_df['Year'] == validationYear]
            xvalidation = validation_df[THERMO + KIN]
            yvalidation = validation_df['Result']
            clf = tuneHyperparameters(xvalidation, yvalidation) 
            
            # We'll have to remove the year we used for validation from the training period
            training_df = training_df[training_df['Year'] != validationYear]
            xtrain = training_df[THERMO + KIN]
            ytrain = training_df['Result']
           

    # Run the model on the data, and collect a number of verification metrics      
    y_pred = clf.predict(xtest)
    accuracy = metrics.accuracy_score(ytest, y_pred)
    yprob = clf.predict_proba(xtest)[0:, 1]
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
    print('Probabilistic Verification')
    print('Brier Score {:3.2f}, '.format(brier_score) + 'AUC = {:3.2f} '.format(auc))
    print('Deterministic Verification')
    print('Accuracy = {:3.2f}, '.format(accuracy) + 'Cohen\'s Kappa {:3.2f}'.format(ck))
    print(cm_df.head())

    # If desired, run stats for the RRA product and a blend of the RRA and the random forest model
    if response6 == '2' or len(response6) == 0:
        # Stats for RRA
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
        print('Probabilistic Verification')
        print('Brier Score {:3.2f}, '.format(brier_score) + 'AUC = {:3.2f} '.format(auc))
        print('Deterministic Verification')
        print('Accuracy = {:3.2f}, '.format(accuracy) + 'Cohen\'s Kappa {:3.2f}'.format(ck))
        print(cm_df.head())

        # Stats for blend of RRA and random forest model
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
        print('Probabilistic Verification')
        print('Brier Score {:3.2f}, '.format(brier_score) + 'AUC = {:3.2f} '.format(auc))
        print('Deterministic Verification')
        print('Accuracy = {:3.2f}, '.format(accuracy) + 'Cohen\'s Kappa {:3.2f}'.format(ck))
        print(cm_df.head())

    print('\n\n')

    # Save to a pickle file if desired
    if response8 == '1':
        pickle.dump(clf, open(saveFile, 'wb'))
    
    # Create plots
    createReliabilityPlots(response5, response6, ytest, yprob, yprob2, yprob3)