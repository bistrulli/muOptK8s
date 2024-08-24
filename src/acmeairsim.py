import numpy as np
from  scipy.io import loadmat
from  scipy.io import savemat
import time
import subprocess
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
import math
from pathlib import Path
import mat73
import logging
import sys
home_dir = Path.home()

logger=None

def setup_custom_logger(name):
    formatter = logging.Formatter(fmt='%(asctime)s %(levelname)-8s %(message)s',
                                  datefmt='%Y-%m-%d %H:%M:%S')
    handler = logging.FileHandler('log.txt', mode='w')
    handler.setFormatter(formatter)
    screen_handler = logging.StreamHandler(stream=sys.stdout)
    screen_handler.setFormatter(formatter)
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.addHandler(handler)
    logger.addHandler(screen_handler)
    return logger

def solveModel(diffLQNPath=f"{home_dir}/DiffLQN_0.1/DiffLQN.jar",modelPath=None):
	logger.info(f"Solving model {modelPath}")
	modelPath=Path(modelPath)
	diffLQNPath=Path(diffLQNPath)
	subprocess.run(["java","-jar",f"{str(diffLQNPath)}",f"{str(modelPath)}"],
					stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT,check=True)

def updateModel(modelPath=None,params={}):
	logger.info(f"Updating model in path {modelPath}")
	modelPath=Path(modelPath)
	lqn=open(modelPath,"r").read()
	for idx,key in enumerate(params):
		lqn=lqn.replace(key,str(params[key])) 
	modelf=open("%s/model.lqn"%(modelPath.parent),"w+")
	modelf.write(lqn)
	modelf.close()

def getResults(resPath=None):
	df=pd.read_csv(resPath,names=["metric","type","name","value"],quotechar='"',
                   skipinitialspace=True, encoding='utf-8')
	return df

def computeCtrl(res=None,tgt=None):
	utils=res[res["metric"]=="utilization"]["value"]
	return np.divide(utils.to_numpy(dtype=float),tgt)

def main():
	global logger
	logger=setup_custom_logger("AcmeairCtrlSim")
	MS=["MSauth","MSvalidateid","MSbookflights","MSupdateMiles","MScancelbooking","MSgetrewardmiles","MSqueryflights",
"MSviewprofile","MSupdateprofile"]
	optTracePath=Path(__file__).parent.parent/Path("re.mat")
	if(optTracePath.exists()):
		data=mat73.loadmat(optTracePath)
	else:
		raise ValueError(f"Optimization Trace {optTracePath} Not Found")
	logger.info("Data Loaded")
	clients=data["Clients"]
	tpred=[]
	for idx,c in enumerate(clients):
		params={}
		logger.info(f"Evaluating case with Clients={int(c)}")
		params["$USERS"]=int(c)
		for ms in MS:
			msidx=MS.index(ms)
			params[f"$Proc{ms}"]=int(np.ceil(data["NC_opt"].T[idx,msidx]))
		updateModel(modelPath=Path(__file__).parent.parent/Path("models/acmeair_tpl.lqn"),params=params)
		solveModel(modelPath=Path(__file__).parent.parent/Path("models/model.lqn"))
		res=getResults(resPath=Path(__file__).parent.parent/Path("models/model.csv"))
		tpred+=[res[(res["metric"]=="throughput")&(res["name"]=="\tclientEntry")]["value"].iloc[0]]
		tm=data["Topt"][idx]
		logger.info(f"{tpred[-1]}--{tm}")

	plt.figure()
	plt.plot(tpred,label="pred")
	plt.plot(data["Topt"],label="mes")
	plt.legend()
	plt.grid()
	plt.show()

if __name__ == '__main__':
	main()