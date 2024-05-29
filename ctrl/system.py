'''
Created on 31 gen 2022

@author: emilio
'''

import matlab.engine
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st
import redis
import subprocess
import time
import json
from threading import Thread, Lock
from scipy.io import savemat
from scipy.io import loadmat
#from generator.SinGen import SinGen

class ThreeTier():
    modelFilePath = None
    modelDirPath = None
    res = None
    matEng = None
    r = None
    Tsim  = None
    w = None
    toStop=None
    state=None
    NC=None
    NT=None
    MU=None
    X=None

    def __init__(self, modelPathStr):
        self.modelFilePath = Path(modelPathStr)
        if(self.modelFilePath.exists()):
            self.modelDirPath = self.modelFilePath.parents[0]
        else:
            raise ValueError("File %s not found" % (modelPathStr)) 
        self.matEng = matlab.engine.start_matlab()
        self.matEng.rng("shuffle","combRecursive")
        self.matEng.cd(str(self.modelDirPath.absolute()))
        self.r=redis.Redis()
        
        self.Tsim=[]
        self.w =[]
        self.toStop=False

        self.NC=np.ones([1,4])*1000
        self.NT=np.ones([1,4])*1000
        self.X=np.zeros([1,14])
        self.MU=np.zeros([1,13])
        self.X[0,0]=1.0

        self.stateLock=Lock()
        
    def simulate(self,X0,NT,NC,MU,dt,TF,nrep=1):
        X = self.matEng.lqn(matlab.double(X0), matlab.double(MU),
                          matlab.double(NT), matlab.double(NC), TF, nrep, dt)
        return np.array(X)
    
    def setUsers(self,W):
        if(W >= self.getUsers()):
            #add W-self.getUsers() users to the system
            X=self.getState()[0,0]+=int(W)-self.getUsers()
            self.setState(X)
        else:
            # %X(0)=XE0_Think;
            # %X(1)=XE0_E0toE1;
            # %X(2)=XE1_a;
            # %X(3)=XE1_E1toE2;
            # %X(4)=XE1_e1Work;
            # %X(5)=XE1_E0toE1;
            # %X(6)=XE2_a;
            # %X(7)=XE2_E2toE3;
            # %X(8)=XE2_e2Work;
            # %X(9)=XE2_E1toE2;
            # %X(10)=XE3_a;
            # %X(11)=XE3_e3Work;
            # %X(12)=XE3_E2toE3;
            removed=0
            while removed<W:
                X=self.getState()
                

            #remove self.w-W users from the system

    def getUsers(self):
        return np.sum(self.X[0,[0,2,4,6,8,10,11]])

    def setState(self,X):
        self.stateLock.acquire()
        self.X=X
        self.stateLock.release()

    def getState(self):
        return self.X
    
    def startSim(self,samplingPeriod):
        dt=1.
        TF=float(samplingPeriod)
        nrep=1
        
        self.MU[0,[0,4,8,11]]=[1.0/1.0299721907508497,1.0/0.05555555555555555,1.0/0.05555555555555555,1.0/0.05555555555555555];
        
        Tstep=[]
        
        while(not self.toStop):
            #simulo il sitema per una passo e salvo lo stato
            #l'aggiornamento del numero di utenti lo faccio alterando lo stato del sistema in modo concorrente
            print(self.X,"-",self.getUsers())
            self.X[0,13]=0.0
            Xi=self.simulate(X0=self.X,NT=self.NT,NC=self.NC,MU=self.MU,
                dt=dt,TF=TF,nrep=1)
            self.setState(Xi[:,-1].reshape(14,1).T)
            #time.sleep(samplingPeriod)
            

if __name__ == '__main__':
    
    # period=400
    # shift=1520
    # mod=1500
    # step=100000
    #sgen=SinGen(period,shift,mod)    
    
    sys=ThreeTier("/Users/emilio-imt/git/MPP4Lqn/model/3tier_k8s/lqn.m")
    steps=[]
    
    #w=W[t]
    #w=sgen.getUser(t)
    #print("############Step=%d#############"%(t))    
    s=sys.startSim(samplingPeriod=1)
    
    # savemat("muopt_%d_%s.mat"%(w,"o"), {"Tsim":sys.Tsim,"NT":np.array(sys.optNT)[:,1:5],"NC":np.array(sys.optNC)[:,1:5],"stimes":sys.stimes,
    #                   "w":sys.w,"steps":steps})