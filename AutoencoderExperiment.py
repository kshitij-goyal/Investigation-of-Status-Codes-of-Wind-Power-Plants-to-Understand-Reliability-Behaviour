# -*- coding: utf-8 -*-

import loadWindTurbineslFromDb as loadDB
from DatenbankAuslesung_Betrieb import db_read as  db_read_Betrieb
import dataPreparation as prepDat
import downRegulationFilter as drf
import anomalyDetectionModels as models
from automaticDataFilter import kFoldIndexes
import pickle
from sklearn.metrics import roc_curve, confusion_matrix, auc
import pandas as pd
import time
import matplotlib.pyplot as plt
import numpy as np
import os.path
from datetime import timedelta 
import tensorflow as tf
import seaborn as sns
import matplotlib.dates as mdates
import datetime as dt
import os
import gc

#%%
def partitionData2017(data,normalIdx):
    numSamples = len(data['X'])
    numFolds = 2
#    trainIdx,testIdx= kFoldIndexes(numSamples,curFoldIdx=(numFolds-1),numFolds=numFolds) #2017-2018
    testIdx,trainIdx= kFoldIndexes(numSamples,curFoldIdx=(numFolds-1),numFolds=numFolds) #2016-2017

    return {'Xtrain':data['X'][trainIdx[normalIdx[trainIdx]],:],
            'ytrain':data['y'][trainIdx[normalIdx[trainIdx]],:],
            'timeTrain':data['timeVec'][trainIdx[normalIdx[trainIdx]],:],
            'normalIdxTrain':normalIdx.reshape(-1,1)[trainIdx[normalIdx[trainIdx]],:],
            'Xtest':data['X'][testIdx,:],
            'ytest':data['y'][testIdx,:],
            'timeTest':data['timeVec'][testIdx,:],
            'normalIdxTest':normalIdx.reshape(-1,1)[testIdx,:]}
def partitionData2016(data,normalIdx):
    numSamples = len(data['X'])
    numFolds = 2
    trainIdx,testIdx= kFoldIndexes(numSamples,curFoldIdx=(numFolds-1),numFolds=numFolds) #2017-2018
#    testIdx,trainIdx= kFoldIndexes(numSamples,curFoldIdx=(numFolds-1),numFolds=numFolds) #2016-2017

    return {'Xtrain':data['X'][trainIdx[normalIdx[trainIdx]],:],
            'ytrain':data['y'][trainIdx[normalIdx[trainIdx]],:],
            'timeTrain':data['timeVec'][trainIdx[normalIdx[trainIdx]],:],
            'normalIdxTrain':normalIdx.reshape(-1,1)[trainIdx[normalIdx[trainIdx]],:],
            'Xtest':data['X'][testIdx,:],
            'ytest':data['y'][testIdx,:],
            'timeTest':data['timeVec'][testIdx,:],
            'normalIdxTest':normalIdx.reshape(-1,1)[testIdx,:]}

def cutTimeRange(dataMat,timeVec,startDate,endDate):
    boolSlice = np.where(np.logical_and(timeVec>=startDate,timeVec<endDate))[0]
    return dataMat[boolSlice,:]
def cutData(data,startDate,endDate):
    data['X'] = cutTimeRange(data['X'],data['timeVec'],startDate,endDate)
    data['y'] = cutTimeRange(data['y'],data['timeVec'],startDate,endDate)
    data['timeVec'] = cutTimeRange(data['timeVec'],data['timeVec'],startDate,endDate)

#%% Plotting functions

def drawRocDiagram(modelNameList,score,normalIdx):
    plt.figure()
    for modelName in modelNameList:
        fpr,tpr,thresholds = roc_curve(normalIdx,score[modelName])
        plt.plot(tpr,fpr,'-',label = modelName)
    plt.plot([0,1],[0,1],'r--')
    plt.xlabel('True Positive Rate')
    plt.ylabel('False Positive Rate')
    plt.xlim([0,1])
    plt.ylim([0,1])
    plt.legend()
    
#%% Call the excel sheet to start running over all the corrective maintenance
ranking=pd.DataFrame(columns=('Name','Value'))

scada=db_read_Betrieb(""" select ot.object_id,t.* from SCADA_EVENTS t left join wind_pool_stamm.wind_turbine_tab wtt on t.wind_turbine_id=wtt.wind_turbine_id left join wind_pool_stamm.object_tab ot on ot.object_id=wtt.object_id  where  t.eventtype=53 and ot.operator_id=1012""").iloc[:,0:6]
scada.columns=['Object_Id','WindT_Id','Start_date','End_date','Cause','Downtime[H]']
scada=scada[(scada['Start_date'] > '2016-02-03') & (scada['Start_date'] < '2018-02-03')] #filter the data to be into the period of data of the autoencoder
scada['4_before']=scada['Start_date']-timedelta(days=4)
scada=scada[:3]
#%%
counter=134
for i,j,k,p in zip(scada['Object_Id'],scada['4_before'],scada['Start_date'],scada['Start_date']):
    
    os.chdir('C:/Users/aortega/info/Master thesis')
    
    
    if __name__=='__main__':
        
        #% Load data section
        # establish connection to the MWABS database
        try:
            con = loadDB.connectToMwabs();
        except:
            con = None    
        # pick turbine identifier
        objectId = int(i)
        # get data from file (if already loaded) or from data base (if not presafed yet)
    #    tic = time.time()
        df = loadDB.fastAccessTrianelDataFrame(con,objectId)
    #    print(time.time()-tic)
        # close connection
        if con!=None:
            con.close()
        # prepare data
        data = prepDat.prepareData(df)
    #    del df #release memory
        
        #%% Cut time range off
        startDate = np.datetime64('2016-02-03')
        endDate = np.datetime64('2018-02-03')
        cutData(data,startDate,endDate)
        
        #%% Filter down regulations and partition data
        # get down regulations
    #    tic = time.time()
        try:
            constIndicator = drf.downRegulationFilter(data,plotFlag=False) #This filter the data when wind is available and power is inyected to grid
            
        except IndexError:
#            counter+=1
            print('no constindicator')
            gc.collect()
            continue
        oo=np.logical_or(data['y']==7,data['y']==4) #this is made to include ready into the data 
#        kk=np.logical_and(~constIndicator,constIndicator)
        # normal index without down regulation or othe operational modes than 7 (normal mode) and 4(ready)
        normalIdx = oo.reshape(-1,) #Filter when there is wind, power production and operational mode=7
        #normal index without down regulation or othe operational modes than 7 (normal mode)
#        normalIdx = np.logical_and(~constIndicator,(oo).reshape(-1,)) #Filter when there is wind, power production and operational mode=7
        
#        normalIdx = oo #Take out the downregulation filter hopefully



        # partition data in train, validation and test
        del constIndicator,oo
        year=str(k).split()[0][0:4]
        print(year)
#        year1=0
        const=0
        if year=='2015':
            plt.clf()   
            plt.close('all')
            gc.collect()
            continue
        elif year=='2016':
            partData = partitionData2016(data,normalIdx) #make the test set to be in 2016
            const=2           
        elif year=='2017':
            partData = partitionData2017(data,normalIdx) #make the test set to be in 2017
            const=4

#        mm=pd.DataFrame(partData['timeTest']) #use to be able to see dates for test set
        
        #%%Block create to avoid problem with date out of the range define by the year in the splitting process
        TimeStampTest=pd.DataFrame(partData['timeTest'],columns=['Date'])
        startDate=(j)
        endDate=(k)

        cnt=0
        cnt2=0 
        while cnt<=300:
            try:
                start_date_num=TimeStampTest[TimeStampTest['Date']==str(startDate)].index.values[0]     
                cnt=305
#                print(cnt)
#                print(start_date_num)
            except IndexError:
#                print(cnt)
                startDate=startDate-timedelta(minutes=10)
                cnt+=1
               
        while cnt2<=300:
            try:         
                end_date_num=TimeStampTest[TimeStampTest['Date']==str(endDate)].index.values[0]
                cnt2=305
#                    print(cnt2)
            except IndexError:           
                endDate=endDate+timedelta(minutes=10)
                cnt2+=1   
                
        startDate=(j)
        endDate=(k)
        
        if cnt!=305 and cnt2!=305:
            const=const+1
#            print('wow')
        if const==3:
            partData = partitionData2017(data,normalIdx)
            year1=str(int(year)+1)
            print('cambio1')
        elif const==5:
            partData = partitionData2016(data,normalIdx)
            year1=str(int(year)-1)
            print('cambio2')
        
        del normalIdx
        TimeStampTest=pd.DataFrame(partData['timeTest'],columns=['Date'])
        cnt=0
        cnt2=0 
        while cnt<=300:
            try:
                start_date_num=TimeStampTest[TimeStampTest['Date']==str(startDate)].index.values[0]     
                cnt=305
#                    print(cnt)
#                print(start_date_num)
            except IndexError:
#                print(cnt)
                startDate=startDate-timedelta(minutes=10)
                cnt+=1
               
        while cnt2<=300:
            try:         
                end_date_num=TimeStampTest[TimeStampTest['Date']==str(endDate)].index.values[0]
                cnt2=305
#                    print(cnt2)
            except IndexError:
#                    print(cnt2)
#                    print('EndDate')               
                endDate=endDate+timedelta(minutes=10)
                cnt2+=1   
                         
        if cnt!=305 or cnt2!=305:
                print('dates not found')
                gc.collect()
                continue
        
        startDate=str(startDate)
        endDate=str(endDate)
#        mm=pd.DataFrame(partData['timeTest'])
#        os.chdir('C:/Users/aortega/info/Master thesis') 
        path='C:/Users/aortega/info/Master thesis/4_Scada_anomalies/Plots_'+str(p)[0:10]+'_'+str(counter)+'_'+str(i)
        try:
            os.mkdir(path)
        except FileExistsError:
            pass
        #%%
        os.chdir('C:/Users/aortega/info/Master thesis') 
        reconTest= dict()
        #%% Train model
        modelNameList = ['UndercompleteAutoencoder_5_']
        
        #%%
        for modelName in modelNameList:
            print(modelName)
            #modelName = 'UndercompleteAutoencoder_10_'
            numEpochs = 20
            if const==3 or const==5:
                fileName = 'reconTest_'+modelName+''+str(objectId)+'_'+year1+'.npy' #Modify because if change of turbine the recontest file need to be changed
            else:
                fileName = 'reconTest_'+modelName+''+str(objectId)+'_'+year+'.npy' #Modify because if change of turbine the recontest file need to be changed

#            fileName = 'reconTest_'+modelName+''+str(objectId)+'_'+year+'.npy' #Modify because if change of turbine the recontest file need to be changed
            if os.path.isfile(fileName):
                reconTest[modelName] = np.load(fileName)
                #reconTest[modelName]=reconTest[modelName][0,:,:]
            else:
                          
                if modelName == 'UndercompleteAutoencoder_5_':
                    modelObj = models.UndercompAutoEnc(numEpochs=numEpochs,
                                                       numUnitsList=[None,5,None], 
                                                       actFuncList=['linear','relu','linear'],
                                                       encodeLayerIdx=1)
          
                modelObj.fit(partData['Xtrain'])
                reconTest[modelName] = modelObj.predict(partData['Xtest'])[0]
                np.save(fileName,reconTest[modelName])
        
        #%% Figure 1 Time Series
        # get wind speed column index
        wsIdx = np.where(data['columnNames']==np.array('WIND_SPEED_CALCULATED'))[0][0]
        # get power column index
        try:
            pwIdx = np.where(data['columnNames']==np.array('ACTIVE_POWER_HV_GRID'))[0][0]
        except IndexError:
            gc.collect()
            continue
        
        # error score: indicating anomalies
        try:
            score = dict()
            for modelName in modelNameList:
                score[modelName] = np.sqrt(np.mean((reconTest[modelName]-partData['Xtest'])**2,axis=1))
        except ValueError:
            print(str(p)+' no se pudo ejecutar')
            gc.collect()
            continue
            
        print(year) 
#        print(type(year))
        #%%
        if year=='2016' or year=='2017': 
#            print('ohhhh')
            os.chdir(path)
            print(path)
            params = {'axes.labelsize': 18,'axes.titlesize':20, }
            plt.rcParams.update(params)
            plt.figure(figsize=(10,10))
            ax = plt.subplot(3,1,3)
            plt.semilogy(partData['timeTest'],score[modelName])
            #plt.ylim([0,5])
            plt.xticks(rotation=45)
            plt.ylabel('Reconstruction Error' )
            #plt.savefig('Power.png',dpi=1000)
            
            plt.subplot(3,1,2,sharex=ax)
            #plt.plot(partData['timeTest'],partData['ytest']==7)
            plt.plot(partData['timeTest'],partData['normalIdxTest'])
            plt.ylabel('Status')
            plt.xticks(rotation=45)
            #plt.savefig('Status.png',dpi=1000)
            
            plt.subplot(3,1,1,sharex=ax)
            plt.plot(partData['timeTest'],data['stats']['std'][pwIdx]*reconTest[modelName][:,pwIdx] + data['stats']['mean'][pwIdx])
            plt.plot(partData['timeTest'],data['stats']['std'][pwIdx]*partData['Xtest'][:,pwIdx] + data['stats']['mean'][pwIdx])
            plt.ylim([0,6000])
            plt.xlim(partData['timeTest'][[5000,6000]])
            plt.ylabel('Power')
            plt.legend(['reconstruction','measurement'])
            
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig('test_'+str(objectId)+'.png',dpi=500,tight=True)
            plt.close('all')
        #    plt.show()
#            del data #Erase data to release computer memory
        #    %%
            
#            jh=np.flip(partData['normalIdxTest'],0)
#            jg=np.flip(partData['normalIdxTest'],0)
            threshold = 0.58
            tpr,fpr,thresholds = roc_curve(partData['normalIdxTest'],score[modelName])
#            fpr,tpr,thresholds = roc_curve(jh,score[modelName])
            idx = np.where(thresholds<threshold)[0]
            thresholdIdx = idx[np.argmax(thresholds[idx])]
            threshold = thresholds[thresholdIdx]
            optThreshold = thresholds[np.argmax((1-tpr)*fpr)];
            auc(fpr,tpr)
            plt.plot(fpr,tpr)
            
            

            for i in range(0,300,1): #Use to set the FPR between 9.5% to 10.5%
                if fpr[thresholdIdx]>0.095 and fpr[thresholdIdx]<0.105:
#                    print(3)
#                    print(fpr[thresholdIdx])
                    break
                elif fpr[thresholdIdx]<0.098:
#                    print(1)
                    threshold = threshold -0.007
                    idx = np.where(thresholds<threshold)[0]
                    thresholdIdx = idx[np.argmax(thresholds[idx])]
                  
#                    print(fpr[thresholdIdx])
                else:
#                    print(2)
                    threshold = threshold +0.007
                    idx = np.where(thresholds<threshold)[0]
                    thresholdIdx = idx[np.argmax(thresholds[idx])]  
#                    print(fpr[thresholdIdx])
            
            threshold = thresholds[thresholdIdx]
            optThreshold = thresholds[np.argmax((1-tpr)*fpr)];
            
            plt.clf()
            plt.figure(figsize=(7,7))
            ax = plt.subplot(2,2,3)
            plt.semilogy(partData['timeTest'],score[modelName])
            plt.semilogy(partData['timeTest'],0*score[modelName]+threshold,'r')
            #plt.ylim([0,5])
            plt.ylabel('Reconstruction Error' )
        
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%d.%m'))
            plt.xticks(rotation=45)
            
            plt.subplot(2,2,1,sharex=ax)
            #plt.plot(partData['timeTest'],partData['ytest']==7)
            plt.plot(partData['timeTest'],partData['normalIdxTest'])
            plt.plot(partData['timeTest'],score[modelName]<threshold)
            plt.ylabel('Status')
            plt.xlim(partData['timeTest'][[5000,6000]])
            plt.legend(['ground truth','predicted anomaly'])
            plt.xticks(rotation=45)
            
            plt.subplot(1,2,2)
            plt.plot(fpr,tpr)
            plt.plot(fpr[thresholdIdx],tpr[thresholdIdx],'ro')
            plt.text(fpr[thresholdIdx]+0.03,tpr[thresholdIdx],s=(str(tpr[thresholdIdx])[:4],str(fpr[thresholdIdx])[:4]),fontsize=4)
            
            plt.plot([0,1],[0,1],'r--')
            plt.ylabel('True Positive Rate')
            plt.xlabel('False Positive Rate')
            plt.xlim([0,1])
            plt.ylim([0,1])
            
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig('test1_'+str(objectId)+'.png',dpi=500)
            plt.clf() 
            plt.close('all')
        #    plt.show()
            
        #    %%
            plt.figure(figsize=(10,14))
            
            plt.subplot2grid((4, 1), (0,0), rowspan=2)
            plt.imshow((reconTest[modelName]-partData['Xtest']).T,aspect='auto',vmin=-10,vmax=10)
            plt.set_cmap('seismic')
            plt.ylabel('Sensor Index')
#            plt.savefig("sensor_index_turbine "+str(objectId)+".png",dpi=100)
            
            plt.subplot2grid((4, 1), (2,0))
            plt.imshow(np.concatenate((partData['normalIdxTest'].reshape(1,-1),(score[modelName]<optThreshold).reshape(1,-1)),axis=0),aspect='auto')
            plt.set_cmap('gray')
            
            plt.subplot2grid((4, 1), (3,0))
            plt.imshow(( partData['ytest']==np.arange(1,15).reshape(1,-1)).T,aspect='auto')
            plt.xlabel('Time Index')
        #    plt.yticks(np.arange(1,16,1))
            plt.set_cmap('Greys')
        #    plt.title('Sensor index and true ground from '+startDate+' until '+endDate+' in turbine '+str(objectId),fontsize=10)
            plt.tight_layout()
            plt.savefig('test2_'+str(objectId)+'.png',dpi=500)
        #    plt.show()
            
            
            plt.clf()   
            plt.close('all')
        #    %% Figure 2 Receiver Operating Characteristic
        #    drawRocDiagram(modelNameList,score,partData['normalIdxTest'])
          
        #    %%Create list of dates of failures starting few dates before to see time horizon of the anomaly
        
        
    #%% Sensor Vs timestamp heatmap
          
            params = {'axes.labelsize': 60,'axes.titlesize':80, }
            plt.rcParams.update(params)
            TimeStampTest=pd.DataFrame(partData['timeTest'],columns=['Date'])
            reconst=pd.DataFrame(reconTest[modelName])     
            reconst.index=TimeStampTest['Date'] #put time like index
            y,x=np.shape(reconst)   
            data_fail=pd.DataFrame(partData['Xtest'])
            data_fail.index=TimeStampTest['Date'] #put time like index  
            difference=pd.DataFrame(index=[np.arange(0,x,1)],columns=[np.arange(0,y,1)])
            difference=reconst-data_fail
            
            del reconTest
            #Here we need to set starting date and ending date to plot
#            startDate=(j)
#            endDate=(k)
#
#            cnt=0
#            cnt2=0 
#            while cnt<=100:
#                try:
#                    start_date_num=TimeStampTest[TimeStampTest['Date']==str(startDate)].index.values[0]     
#                    cnt=101
##                    print(cnt)
##                    print('StartDate')
#                except IndexError:
##                    print(cnt)
#                    startDate=startDate-timedelta(seconds=600)
#                    cnt+=1
#                   
#            while cnt2<=100:
#                try:         
#                    end_date_num=TimeStampTest[TimeStampTest['Date']==str(endDate)].index.values[0]
#                    cnt2=101
##                    print(cnt2)
#                except IndexError:
##                    print(cnt2)
##                    print('EndDate')               
#                    endDate=endDate+timedelta(seconds=600)
#                    cnt2+=1   
#                    
#            startDate=str(startDate)
#            endDate=str(endDate)        
                    
            try:        
                fig, ax=plt.subplots(figsize=(110,50))
                plt.imshow((difference[startDate:endDate]).T,aspect='auto',vmin=-10,vmax=10)
                plt.set_cmap('seismic')        
                plt.ylabel('Sensor Index')
                plt.xlabel('TimeStamp')
                xlabels=pd.DataFrame(difference[startDate:endDate].index[:])
                ax.set_xticks(np.arange(len(difference[startDate:endDate])))
                del difference, data_fail
                ax.set_xticklabels((xlabels['Date']))
                plt.xticks(rotation=-90)
                plt.yticks(np.arange(0,x,2))
                
            #    plt.ylim([sensor_ini,sensor_end])
                plt.title('Anomalies from '+startDate+' until '+endDate+' in turbine '+str(objectId))
                plt.savefig('Anomalies_'+str(objectId)+'.png',pdi=300) 
                plt.clf()   
                plt.close('all')
                
            
                #Cumulative plot
                fig, ax=plt.subplots(figsize=(110,20))
                
                ll=np.concatenate((partData['normalIdxTest'].reshape(1,-1),(score[modelName]<optThreshold).reshape(1,-1)),axis=0).T[start_date_num:end_date_num]         
                plotCounter=pd.DataFrame(index=range(start_date_num,end_date_num,1))
                conter=start_date_num 
                cum=0
                for d,f in ll:
                    if f==False and d==True and cum>=0:
                        plotCounter.loc[conter,'Num']=1
                    else:
                        plotCounter.loc[conter,'Num']=0
                    conter+=1
                
                cont2=start_date_num
                for e in plotCounter['Num']:
                    if cum>0 and e==0:
                        cum=cum-1
                        plotCounter.loc[cont2,'Cum']=cum
                        cont2+=1
                    elif cum==0 and e==0:
                        cum=0
                        plotCounter.loc[cont2,'Cum']=cum
                        cont2+=1
                    elif cum>=0 and e==1:
                        cum=cum+1
                        plotCounter.loc[cont2,'Cum']=cum
                        cont2+=1
                del ll
                ranking.loc[counter,'Name']=p
                ranking.loc[counter,'Value']=np.max(plotCounter['Cum'])
                ax.plot(plotCounter.index,plotCounter['Cum'],lw=8)
                ax.grid(True)
#                ax.axvline(x=end_date_num,color='red')
                ax.set_yticks(np.arange(0,max(plotCounter['Cum'])+2,1))
                ax.tick_params(axis='y', which='major', labelsize=30)
                ax.set_xticks(np.arange(start_date_num,end_date_num,1))
                ax.set_xticklabels(TimeStampTest[start_date_num:end_date_num]['Date'])
                plt.xticks(rotation=-90)
                plt.savefig('Cumulative_plot_'+str(objectId)+'.png',pdi=200) 
                plt.clf()   
                plt.close('all')
                
                #    %% Sensor Vs timestamp heatmap
                fig, ax=plt.subplots(figsize=(110,20))
                plt.imshow(np.concatenate((partData['normalIdxTest'].reshape(1,-1),(score[modelName]<optThreshold).reshape(1,-1)),axis=0),aspect='auto')
                plt.set_cmap('gray')
                ax.set_xlabel('Time Index')
#                ax.axvline(x=end_date_num,color='red')
                plt.yticks(fontsize=20)
                plt.xticks(rotation=-90)
                plt.xlim([start_date_num,end_date_num])
                ax.set_xticks(np.arange(start_date_num,end_date_num,1))
                ax.set_xticklabels(TimeStampTest[start_date_num:end_date_num]['Date'])
                plt.title('Ground_True from '+startDate+' until '+endDate)
                plt.savefig('Ground True_'+str(objectId)+'.png',pdi=200) 
                plt.clf()   
                plt.close('all')
                
            #    %% Sensor vs operation mode
                
                fig, ax=plt.subplots(figsize=(110,20))
                plt.imshow(( partData['ytest']==np.arange(1,15).reshape(1,-1)).T,aspect='auto')
                ax.set_xlabel('Time Index')
#                ax.axvline(x=end_date_num,color='red')
                ax.set_ylabel('Operation mode')
                ax.set_yticks(np.arange(0,16,1))
                Status_codes=['Error','Power outage','Initial','Ready','Boot procedure',
                              'Emergency power UW','Production','Shadowing','Build-up of ice',
                              'Setup','System check','Wind cut-off','Service',
                              'Manual stop','Emergency power WTG']
                ax.set_yticklabels(Status_codes,fontsize=20)
                plt.title('Operational modes from '+startDate+' until '+endDate+' in turbine '+str(objectId))
                plt.xlim([start_date_num,end_date_num])
                ax.set_xticks(np.arange(start_date_num,end_date_num,1))
                ax.set_xticklabels(TimeStampTest[start_date_num:end_date_num]['Date'])
                plt.xticks(rotation=-90)
                plt.set_cmap('Greys')
                plt.savefig('Operational_modes_turbine '+str(objectId)+'.png',dpi=200)
                plt.clf()   
                plt.close('all')
                gc.collect()
                try:
                    del start_date_num, end_date_num, year,year1, TimeStampTest, partData
                except NameError:
                    gc.collect()
                    pass
                ranking.to_excel('ranking until'+str(objectId)+'.xlsx')
            except KeyError:
                print('Dates not found')
                plt.clf()   
                plt.close('all')
                gc.collect()
                
        counter+=1
        gc.collect()
        

