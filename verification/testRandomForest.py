#!/usr/bin/python3

import pickle 
import argparse
import sklearn.ensemble as ens
import numpy as np
import csv


def parse_args(args=None):
    """Parse inputs to script"""
    parser = argparse.ArgumentParser(
        description=__doc__,
        fromfile_prefix_chars='@',
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--filename",
        type=str, required=True,
        help="random forest model filename")
    parser.add_argument(
        "--parameters",
        type=float, nargs='+', required=False,
        help="""Array of atmospheric parameters must be in following order:
             MUCAPE, DCAPE, 7-5LR, LCL, PW, MeanRH,DD, WC, SfcTd, SfcT,
             0-6km Shear Dir, 0-6km Shear Mag, C Dn Dir,
             C Dn Mag, C Up Dir, C Up Mag, SM Dir, SM Mag,
             MW Dir,MW Mag, 4-6km dir,4-6km mag""")
    parser.add_argument(
        "--inputFile",
        type=str, required=True,
        help="Path to input file")
    parser.add_argument(
        "--outputFile",
        type=str, required=False,
        help="Path to output file")	
    return parser.parse_args(args)

def main():
    
    """ Note if you want to retrain the model the easiest way is to just
    follow the line of logic in the python notebooks, this script assumes
    you just have an already saved random forest you just want to make a 
    prediction on with new data, note the array of data you give to the forest
    MUST be in the order listed in the argparse method"""


    args = parse_args()

    filename = args.filename
    rf = pickle.load(open(filename, 'rb'))

    if args.parameters is not None:
        parameters = np.array(args.parameters).reshape(1,-1)
    else:
        parameters = []
        with open(args.inputFile) as f:
          lines = [line.split() for line in f]
          data = lines[1:]
          parms = []
          for date in data:
              entries = []		  
              for entry in date:
                  entries.append(entry.strip(','))
              entries.pop(0)
              parameters.append(np.array(entries).reshape(1,-1))	      		  
    if len(np.shape(parameters)) > 2:
        output = ""
        for date in parameters:		  	      
            probs = rf.predict_proba(date[:]) 
            print ('The predicted probability of flooding is {:3.2f}'.format(probs[0,1]))
            output = output + '{:3.2f}'.format(probs[0,1]) + '\n'
        if args.outputFile is not None:
            f = open(args.outputFile, 'w')
            f.write(output)
            f.close()
    else:	   	         
        probs = rf.predict_proba(parameters[:]) 
        print ('The predicted probability of flooding is {:3.2f}'.format(probs[0,1]))
        if args.outputFile is not None:
            f = open(args.outputFile, 'w')
            f.write('{:3.2f}'.format(probs[0,1]))
            f.close() 

if __name__ == '__main__':
    main()    
