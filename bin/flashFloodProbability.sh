#!/bin/sh

# Darren Van Cleave
# May 9 2019
# Creates text files containing flash flood probability from Random Forest model for flash flooding

/apps/2018FFW_study/soundingAnalysis/sounding_analysis.py --date_list `date +%Y%m%d` `date +%Y%m%d -d '+1 days'` `date +%Y%m%d -d '+2 days'`

/apps/2018FFW_study/verification/testRandomForest.py --filename=/apps/2018FFW_study/Random\ Forest/savedModel_KBCE.pck --inputFile=/apps/2018FFW_study/soundingAnalysis/KBCE.csv --outputFile=/apps/2018FFW_study/output/KBCE.txt

/apps/2018FFW_study/verification/testRandomForest.py --filename=/apps/2018FFW_study/Random\ Forest/savedModel_KSGU.pck --inputFile=/apps/2018FFW_study/soundingAnalysis/KSGU.csv --outputFile=/apps/2018FFW_study/output/KSGU.txt

/apps/2018FFW_study/verification/testRandomForest.py --filename=/apps/2018FFW_study/Random\ Forest/savedModel_KPGA.pck --inputFile=/apps/2018FFW_study/soundingAnalysis/KPGA.csv --outputFile=/apps/2018FFW_study/output/KPGA.txt

/apps/2018FFW_study/verification/testRandomForest.py --filename=/apps/2018FFW_study/Random\ Forest/savedModel_K4HV.pck --inputFile=/apps/2018FFW_study/soundingAnalysis/4HV.csv --outputFile=/apps/2018FFW_study/output/4HV.txt

cd /apps/2018FFW_study/output

  echo "" >> KBCE.txt
  echo "" >> KBCE.txt

  echo "" >> KSGU.txt
  echo "" >> KSGU.txt

  echo "" >> KPGA.txt
  echo "" >> KPGA.txt

  echo "" >> 4HV.txt
  echo "" >> 4HV.txt


PROD=/apps/2018FFW_study/output/SLCRRXSLC

DAY1=`date +%Y%m%d`
DAY2=`date -d '+1 day' +%Y%m%d`
DAY3=`date -d '+2 day' +%Y%m%d`

BCE_VAL1=`head -1 KBCE.txt`
BCE_VAL2=`head -2 KBCE.txt | tail -1`
BCE_VAL3=`head -3 KBCE.txt | tail -1` 

SGU_VAL1=`head -1 KSGU.txt`
SGU_VAL2=`head -2 KSGU.txt | tail -1`
SGU_VAL3=`head -3 KSGU.txt | tail -1` 

PGA_VAL1=`head -1 KPGA.txt`
PGA_VAL2=`head -2 KPGA.txt | tail -1`
PGA_VAL3=`head -3 KPGA.txt | tail -1` 

K4HV_VAL1=`head -1 4HV.txt`
K4HV_VAL2=`head -2 4HV.txt | tail -1`
K4HV_VAL3=`head -3 4HV.txt | tail -1` 

echo "KBCE" > $PROD
  echo "$DAY1, $BCE_VAL1" >> $PROD
  echo "$DAY2, $BCE_VAL2" >> $PROD
  echo "$DAY3, $BCE_VAL3" >> $PROD
echo "KSGU" >> $PROD
  echo "$DAY1, $SGU_VAL1" >> $PROD
  echo "$DAY2, $SGU_VAL2" >> $PROD
  echo "$DAY3, $SGU_VAL3" >> $PROD
echo "KPGA" >> $PROD
  echo "$DAY1, $PGA_VAL1" >> $PROD
  echo "$DAY2, $PGA_VAL2" >> $PROD
  echo "$DAY3, $PGA_VAL3" >> $PROD
echo "4HV" >> $PROD
  echo "$DAY1, $K4HV_VAL1" >> $PROD
  echo "$DAY2, $K4HV_VAL2" >> $PROD
  echo "$DAY3, $K4HV_VAL3" >> $PROD

scp -q $PROD ldad@ls1:/data/Incoming/SLCRRXSLC
