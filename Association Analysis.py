# -*- coding: utf-8 -*-
"""
Created on Mon 15 Dec 15:41:21 2017

@author: kgoyal
"""
from func_warenkorbanalyse_v_2 import (func_warenkorbanalyse_getlist,
                                       func_warenkorbanalyse_listtocsv,
                                       func_warenkorbanalyse_loadlist,
                                       func_heatmap_warenkorbanalyse, func_gephi_table)

from apriori import (
    getItemSetTransactionList,
    dataFromFile,
    joinSet,
    printResults,
    returnItemsWithMinSupport,
    runApriori,
    subsets)


import seaborn as sns; sns.set()
import matplotlib.pyplot as plt
from py_db import db_connect
import cx_Oracle
import os, sys
import numpy as np
import pandas as pd
import math
import pickle
import datetime
from DatenbankAuslesung_Ereignis import db_read

#%% setting up all relevant values
maxFailure=2400000000            #hours
minFailure=0              #hours
maxYearOfEvent=2020          #whole year is included
minDateOfInstallation=1990  #whole year is included
#objects=db_read("""  select t.object_id from WT_STAT_GT1 t  group by t.object_id """)    #manuelle Eingabe erforderlich

#objects=db_read("""  select ot.object_id from GEO_OPERATOR t   left join wind_pool_stamm.object_tab ot on TO_CHAR(t.serialnumber)=TO_CHAR(ot.object_op_id)  where ot.object_id is not null and to_char(ot.date_of_installation,'YYYY') >"""+str(minDateOfInstallation)+"""  group by ot.object_id  order by ot.object_id  asc """)    #manuelle Eingabe erforderlich
objects=db_read(""" select t.object_id from WT_STAT_WSB t where t.object_id is not null  group by t.object_id  order by t.object_id  asc """) 
toleranceRange=24*30        #hours                    #manuelle Eingabe erforderlich     #
minSupport = 0.000001                       #0.01                     #manuelle Eingabe erforderlich
minConfidence = 0.1                         #0.5                #manuelle Eingabe erforderlich
cut_df=1000                                                         #manuelle Eingabe erforderlich #50
#%% end of setting up all relevant values

#%% _getting the list of relevant statuscodes with respect of the time bound
output_array_all_turs, output_array_all_tursstart, output_array_all_tursend=func_warenkorbanalyse_getlist(toleranceRange,objects,minDateOfInstallation,maxYearOfEvent,maxFailure,minFailure)   
#%% end of getting the list of relevant statuscodes with respect of the time bound

######output_array_all_turs, output_array_all_tursstart, output_array_all_tursend=func_warenkorbanalyse_loadlist()

#%% making it look nicely
filestring='statuscodes_toleranceRange='+str(toleranceRange)+'.csv'
csv_data=func_warenkorbanalyse_listtocsv(output_array_all_turs,filestring)

csv_data=csv_data.iloc[:,:cut_df]                                  

max_col_ind=0
for row_ind in range(len(csv_data)):
    for col_ind in range(len(csv_data.iloc[0,:])):
        if type(csv_data.iloc[row_ind,col_ind])==str:
            if col_ind>max_col_ind:
                max_col_ind=col_ind
            #csv_data.iloc[row_ind,col_ind]

csv_data=csv_data.iloc[:,:max_col_ind]  

if max_col_ind==cut_df or max_col_ind==cut_df-1:
    print('Value for cutting DataFrame should be higher')

for row_ind in range(len(csv_data)):
    for col_ind in range(len(csv_data.iloc[0,:])):
        if type(csv_data.iloc[row_ind,col_ind])==float:
            csv_data.iloc[row_ind,col_ind]=float('NaN')

csv_data.to_csv('statuscodes_toleranceRange='+str(toleranceRange)+'.csv',header=False,index=False) 
#%% _end of making it look nicely


#%% start o the apriori algorythm
inFile = dataFromFile('statuscodes_toleranceRange='+str(toleranceRange)+'.csv')
#inFile = dataFromFile('statuscodes.csv')

items, rules = runApriori(inFile, minSupport, minConfidence)
#%% end o the apriori algorythm

#items=tempitems[:64]
#rules=temprules[:70]

#%% saving all the rules to a .txt file
list_of_tuples = rules
f = open('rules_toleranceRange='+str(toleranceRange)+'_MinSupport='+str(minSupport)+'_MinConfidence='+str(minConfidence)+'_maxFailure='+str(maxFailure)+'_minFailure='+str(minFailure)+'.txt', 'w')
for t in list_of_tuples:
    line = ' '.join(str(x) for x in t)
    f.write(line + '\n')
f.close()

list_of_tuples = items
f = open('items_toleranceRange='+str(toleranceRange)+'_MinSupport='+str(minSupport)+'_MinConfidence='+str(minConfidence)+'_maxFailure='+str(maxFailure)+'_minFailure='+str(minFailure)+'.txt', 'w')
for t in list_of_tuples:
    line = ' '.join(str(x) for x in t)
    f.write(line + '\n')
f.close()
#%% end saving all the rules to a .txt file

node_table,edge_table=func_gephi_table()
for a in range(len(items)): 
    node_table=node_table.append(pd.DataFrame([[a,items[a][0],0]], columns=node_table.columns))
node_table=node_table.dropna()

dict_node_table=node_table.set_index('Label')['Id'].to_dict() 

for a in range(len(rules)): 
    try:
        edge_table=edge_table.append(pd.DataFrame([[dict_node_table[rules[a][0][0]],dict_node_table[rules[a][0][1]],0,rules[a][1]]], columns=edge_table.columns))     
    except KeyError:
        pass
edge_table.dropna()

edge_table.to_csv('edge_table.csv',index=False)
node_table.to_csv('node_table.csv',index=False)

#%% plot of heatmap
#if len(list_of_tuples)>0:
#    func_heatmap_warenkorbanalyse(list_of_tuples,minSupport,minConfidence,toleranceRange)
#else:
#    print('No Confidence values to show')
#%% end
