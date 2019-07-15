# -*- coding: utf-8 -*-
"""
Created on Tue Nov 4 10:18:08 2017

@author: kgoyal
"""


     
import numpy as np
import matplotlib.pyplot as plt
from py_db import db_connect
import cx_Oracle
import os, sys
import numpy as np
import matplotlib
import matplotlib.pyplot as pyplot
import openpyxl
import xlrd
import pandas as pd
import math
from DatenbankAuslesung_Betrieb import db_read
from itertools import combinations
import pickle
import datetime
import numpy as np; np.random.seed(0)
import seaborn as sns; sns.set()


def func_warenkorbanalyse_getlist(timebound,objects,minDateOfInstallation,maxYearOfEvent,maxFailure,minFailure):
    
    objects=objects
    #objects=pd.DataFrame([18,19,20,22,23,24])
    
    output_array_all_turs=[None]*len(objects)
    output_array_all_tursstart=[None]*len(objects)
    output_array_all_tursend=[None]*len(objects)
    delta=datetime.timedelta(hours=24)
    
    
    for iter_objects in range(len(objects)):
#        
#        singleobj=db_read("""select * from ((select t.beginofevent,t.endofevent,(t.statusoverview)  from GEO_OPERATOR t where t.endofevent is not null and  t.object_id=""" +str(objects.iloc[iter_objects,0])+""" and to_char(t.beginofevent,'YYYY')> """+str(minDateOfInstallation)+""" and  to_char(t.endofevent,'YYYY')< """+str(maxYearOfEvent+1)+""" and (t.endofevent-t.beginofevent)*24 >="""+str(minFailure)+"""  and (t.endofevent-t.beginofevent)*24<="""+str(maxFailure)+""") union all  (
#                select t.abschaltzeit,t.anschaltzeit,TO_CHAR(t.eventtype)  from SCADA_EVENTS t 
#                                                            left join wind_pool_stamm.wind_turbine_tab wtt on t.wind_turbine_id=wtt.wind_turbine_id
#                                                            left join wind_pool_stamm.object_tab ot on wtt.object_id=ot.object_id
#                                                            where ot.operator_id=1000  and  t.eventtype in (51,52,53)
#                                                            )) datatest order by datatest.beginofevent asc  """)
#        
        singleobj=db_read("""select * from ((select t.beginofevent,t.endofevent,(t.statusoverview)  from GEO_OPERATOR t left join wind_pool_stamm.object_tab ot on TO_CHAR(t.serialnumber)=TO_CHAR(ot.object_op_id) where t.endofevent is not null and  ot.object_id=""" +str(objects.iloc[iter_objects,0])+""" and to_char(t.beginofevent,'YYYY')> """+str(minDateOfInstallation)+""" and  to_char(t.endofevent,'YYYY')< """+str(maxYearOfEvent+1)+""" and (t.endofevent-t.beginofevent)*24 >="""+str(minFailure)+"""  and (t.endofevent-t.beginofevent)*24<="""+str(maxFailure)+""") union all  (
                select t.abschaltzeit,t.anschaltzeit,TO_CHAR(t.eventtype)  from SCADA_EVENTS t 
                                                    left join wind_pool_stamm.wind_turbine_tab wtt on t.wind_turbine_id=wtt.wind_turbine_id
                                                    left join wind_pool_stamm.object_tab ot on wtt.object_id=ot.object_id
                                                    where t.eventtype in (51,52,53) and  ot.object_id=""" +str(objects.iloc[iter_objects,0])+""" 
                                                    )) datatest order by datatest.beginofevent asc  """)

                                                  
        obj_output=pd.DataFrame(np.zeros((len(singleobj),(len(singleobj)))))
        obj_output_starttime=pd.DataFrame(np.zeros((len(singleobj),(len(singleobj)))))
        obj_output_endtime=pd.DataFrame(np.zeros((len(singleobj),(len(singleobj)))))
    
        flag_alltogether=0
        flag_alltogether_laststep=0
        cnt=0
        row_ind=0
        col_ind=0
        flag_firstrow=0    
                                                       
        for iter_singelobj in range(len(singleobj)):
    
            if iter_singelobj>0:
                if singleobj.iloc[iter_singelobj,0]<=singleobj.iloc[iter_singelobj-1,1]+delta and singleobj.iloc[iter_singelobj,0]>=singleobj.iloc[iter_singelobj-1,0]:
                    flag_alltogether=1
                    if flag_firstrow==0:   
                        obj_output.iloc[row_ind,col_ind]=singleobj.iloc[iter_singelobj-1,2]
                        obj_output_starttime.iloc[row_ind,col_ind]=singleobj.iloc[iter_singelobj-1,0]
                        obj_output_endtime.iloc[row_ind,col_ind]=singleobj.iloc[iter_singelobj-1,1]
                        
                        flag_firstrow=1
                    obj_output.iloc[row_ind+1,col_ind]=singleobj.iloc[iter_singelobj,2]
                    obj_output_starttime.iloc[row_ind+1,col_ind]=singleobj.iloc[iter_singelobj,0]
                    obj_output_endtime.iloc[row_ind+1,col_ind]=singleobj.iloc[iter_singelobj,1]
                    
                    row_ind=row_ind+1
                else:
                    flag_alltogether=0
                    row_ind=0
                        
                if  flag_alltogether==0 and flag_alltogether_laststep==1:
                    col_ind=col_ind+1
                    flag_firstrow=0
                    
                flag_alltogether_laststep=flag_alltogether        
         
    
        row_ind=0
        col_ind=0  
        if len(obj_output)>0:
            for a in range(len(obj_output.iloc[0,:])):
                if obj_output.iloc[0,a]==0 and col_ind==0:
                    col_ind=a
            obj_output=obj_output.iloc[:,:col_ind]         
            output_array_all_turs[iter_objects]=obj_output
        else:
            output_array_all_turs[iter_objects]=pd.DataFrame(np.zeros((1,1)))
                                     
                             
        row_ind=0
        col_ind=0  
        if len(obj_output_starttime)>0:
            for a in range(len(obj_output_starttime.iloc[0,:])):
                if obj_output_starttime.iloc[0,a]==0 and col_ind==0:
                    col_ind=a
            obj_output_starttime=obj_output_starttime.iloc[:,:col_ind]                          
            output_array_all_tursstart[iter_objects]=obj_output_starttime
        else:
            output_array_all_tursstart[iter_objects]=pd.DataFrame(np.zeros((1,1)))
                              
                                  
        row_ind=0
        col_ind=0  
        if len(obj_output_endtime)>0:
            for a in range(len(obj_output_endtime.iloc[0,:])):
                if obj_output_endtime.iloc[0,a]==0 and col_ind==0:
                    col_ind=a
            obj_output_endtime=obj_output_endtime.iloc[:,:col_ind]                               
            output_array_all_tursend[iter_objects]=obj_output_endtime           
        else:
            output_array_all_tursend[iter_objects]=pd.DataFrame(np.zeros((1,1)))
            
    
    
    #
    #maxrowind=0
    #for a in range(len(output_array_all_turs)):
    #    temp=output_array_all_turs[a]
    #    for b in range(len(temp.iloc[0,:])):
    #        for c in range(len(temp)):
    #            if type (temp.iloc[c,b]) is str:
    #                if c>maxrowind:
    #                    maxrowind=c+1
    #    temp=temp.iloc[:maxrowind,:]       
    #    output_array_all_turs[a]=temp
    #                         
    #    temp=output_array_all_tursstart[a]     
    #    temp=temp.iloc[:maxrowind,:]
    #    output_array_all_tursstart[a]=temp
    #                              
    #    temp=output_array_all_tursend[a]     
    #    temp=temp.iloc[:maxrowind,:]
    #    output_array_all_tursend[a]=temp
    
                              
                                  
    with open("warenkorb_statuscodes.txt", "wb") as fp:   #Pickling
       pickle.dump(output_array_all_turs, fp) 
    
    with open("warenkorb_starttime.txt", "wb") as fp:   #Pickling
       pickle.dump(output_array_all_tursstart, fp) 
    
    with open("warenkorb_endtime.txt", "wb") as fp:   #Pickling
       pickle.dump(output_array_all_tursend, fp) 
    
    return output_array_all_turs, output_array_all_tursstart, output_array_all_tursend




def func_warenkorbanalyse_listtocsv(code,filestring):
    
    flag_csv=0
    
    for iter_list in range(len(code)):
        temp=code[iter_list]
        temp=temp.transpose()
        #code[iter_list]=temp
        
        if flag_csv==0:
            csv_data=temp
            flag_csv=1
        else:
            csv_data=csv_data.append(temp)
                
        
    csv_data.to_csv(str(filestring),header=False,index=False)    
        
    return csv_data
        
        
def func_warenkorbanalyse_loadlist():

    with open("warenkorb_statuscodes.txt", "rb") as fp:   # Unpickling
       output_array_all_turs = pickle.load(fp)
      
    with open("warenkorb_starttime.txt", "rb") as fp:   
       output_array_all_tursstart= pickle.load(fp) 
    
    with open("warenkorb_endtime.txt", "rb") as fp:   
       output_array_all_tursend= pickle.load(fp) 
   
    return output_array_all_turs, output_array_all_tursstart, output_array_all_tursend
        



def func_heatmap_warenkorbanalyse(list_of_tuples,minSupport,minConfidence,toleranceRange):
            
    index=pd.DataFrame(np.zeros((len(list_of_tuples),1)))
    for iter_ind in range(len(list_of_tuples)):
        temp=list_of_tuples[iter_ind]
        index.iloc[iter_ind,0]=str(temp[0][0])
    
    
    columns=pd.DataFrame(np.zeros((len(list_of_tuples),1)))
    for iter_ind in range(len(list_of_tuples)):
        temp=list_of_tuples[iter_ind]
        columns.iloc[iter_ind,0]=str(temp[0][1])
    
    
    
    uniform_data=pd.DataFrame(np.zeros((len(columns),len(columns))),index=index,columns=columns)*float('NaN')
    
    
    for iter_ind in range(len(uniform_data)):
        for col_ind in range(len(uniform_data)):
            if iter_ind==col_ind:
                temp=list_of_tuples[iter_ind]
                uniform_data.iloc[iter_ind,col_ind]=(temp[1])
    
    
    
    cmap = plt.cm.Reds
    ax = sns.heatmap(uniform_data,cmap=cmap)      
    
    plt.title('Confidence of Statuscodes\n MinSupport='+str(minSupport)+'  MinConfidence='+str(minConfidence)+' \n ToleranceRange='+str(toleranceRange)+'\n', size=16)
    plt.savefig('Heatmap_assoziationsanalyse.png', dpi=1000,bbox_inches='tight')         
    plt.show()        
            
def func_gephi_table():
    node_table=pd.DataFrame(np.zeros((1,3)),columns=['Id','Label','Type'])*float('NaN')
    edge_table=pd.DataFrame(np.zeros((1,4)),columns=['Source','Target','Label','Weight'])*float('NaN')      
    return node_table, edge_table
            
        