import time
import subprocess
import numpy as np
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
from scipy.io import savemat
import math


def solveModel(diffLQNPath="/Users/emilio-imt/DiffLQN_0.1/DiffLQN.jar",modelPath=None):
    modelPath=Path(modelPath)
    diffLQNPath=Path(diffLQNPath)
    subprocess.run(["java","-jar",f"{str(diffLQNPath)}",f"{str(modelPath)}"],
                    stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT,check=True)

def updateModel(modelPath=None,params={}):
    modelPath=Path(modelPath)
    lqn=open(modelPath,"r").read()
    for idx,key in enumerate(params):
        lqn=lqn.replace(key,str(params[key])) 
    modelf=open("%s/model.lqn"%(modelPath.parent),"w+")
    modelf.write(lqn)
    modelf.close()

def getResults(resPath=None):
    df=pd.read_csv(resPath,names=["metric","type","name","value"])
    return df

def computeCtrl(res=None,tgt=None):
    utils=res[res["metric"]=="utilization"]["value"]
    return np.divide(utils.to_numpy(dtype=float),tgt)

def getSimTrace():
    simD=np.random.randint(10,300,10)
    sim=[]
    for i in range(simD.shape[0]):
        sim+=[simD[i]]*5
    return sim

def sinTrace():
    mod = 200
    shift = 10
    period = 60 / (2*math.pi)
    sim=[]
    
    x=np.linspace(0,120,121)
    for t in x:
        sim+=[np.abs(math.sin(t/period)*mod)+shift]
    
    return sim

def main():
    #genero una carico
    #per ogni punto, risolvo gli utilizzi e calcolo il numero di repliche in base al target
    #mi salvo gliutilizzi misurati
    pass

if __name__ == '__main__':
    sim=sinTrace()
    
    utils=[]
    corshist=[]
    ncores=np.array([1,1,1])
    #e=np.maximum(np.random.rand(10),0.005)*5
    e=[0.1,0.1,0.1]
    for u in tqdm(sim):
        corshist+=[ncores]
        params={"$r1":int(ncores[0]),
                "$r2":int(ncores[1]),
                "$r3":int(ncores[2]),
                "$users":int(u),
                "$e1":e[0],
                "$e2":e[1],
                "$e3":e[2]
                }
        updateModel(modelPath="../models/3tier_tpl.lqn",params=params)
        solveModel(modelPath="../models/model.lqn")
        res=getResults(resPath="../models/model.csv")
        utils.append(np.divide(res[res["metric"]=="utilization"]["value"].to_numpy(dtype=float),ncores).tolist())
        ncores=np.maximum(np.ceil(computeCtrl(res=res,tgt=ncores*0.5)*ncores),1)
    
    savemat("trace.mat",{"users":sim})
    
    #print(np.mean(np.array(utils),axis=1))
    x=np.linspace(1,len(sim),len(sim))
    # Create two subplots and unpack the output array immediately
    f, (ax1, ax2,ax3) = plt.subplots(3, 1,sharex=True)
    ax1.step(x,np.array(utils))
    ax1.axhline(y=0.5,linewidth=1, color='r')
    ax1.grid()
    ax1.set_ylabel("U")
    ax2.step(x,sim)
    ax2.grid()
    ax2.set_ylabel("#Users")
    ax3.step(x,np.array(corshist))
    ax3.grid()
    ax3.set_ylabel("#Repliche")
    plt.show()
        