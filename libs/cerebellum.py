import numpy as np
from json import dump, loads


class Cerebellum:

    def micro_circuit(self, idof, thisInput) :
           
        self.wChange[idof]-=self.beta*self.error[idof]*self.bufferOut[idof][self.index,:]
        
        self.excFirstInt[idof]=self.excFactR[idof]*self.excFirstInt[idof]+(1-self.excFactR[idof])*thisInput
        self.excSecInt[idof]=self.excFactD[idof]*self.excSecInt[idof]+(1-self.excFactD[idof])*self.excFirstInt[idof]

        self.inhFirstInt[idof]=self.inhFactR[idof]*self.inhFirstInt[idof]+(1-self.inhFactR[idof])*thisInput
        self.inhSecInt[idof]=self.inhFactD[idof]*self.inhSecInt[idof]+(1-self.inhFactD[idof])*self.inhFirstInt[idof]

        exc = np.maximum(self.excSecInt[idof]-self.excThreshold[idof],0)*self.excScaleBasis[idof]
        inh = np.maximum(self.inhSecInt[idof]-self.inhThreshold[idof],0)*self.inhScaleBasis[idof]

        self.bufferOut[idof][self.index,:] = np.maximum(exc-inh,0.0)

        self.out[idof] = self.bufferOut[idof][self.index,:]*self.w[idof]
        self.act[idof] += self.bufferOut[idof][self.index,:]

        if self.save_bases & (self.cTrial==0) & (idof==0):
            self.debug_data['basis']['tot'].append(np.maximum(exc-inh,0.0).tolist())
            self.debug_data['basis']['exc'].append(exc.tolist())
            self.debug_data['basis']['inh'].append(inh.tolist())


    def doUpdate(self):

        timeSinceUpdate = (self.cCounter - self.lastUpdate)/self.SR
        self.lastUpdate = self.cCounter

        for i in range(self.nDoF):
            self.act[i] = self.act[i]**2 # normalizing weight change to square of basis activity
            
            # act>0: only adapt basis that had some activity 
            # 0.01*np.mean(self.act[i][self.act[i]>0] : regularizing, to avoid that very small activities lead to very big changes

            self.w[i][self.act[i]>0]-=self.wChange[i][self.act[i]>0]/(\
                self.act[i][self.act[i]>0]+0.01*np.mean(self.act[i][self.act[i]>0])\
                )*timeSinceUpdate*(1-self.betaInertia)
            self.act[i] *= 0
            self.wChange[i] *= self.betaInertia



    ########################################################
    def saveMemory(self, memoryFileName):

        m={}

        m['ws'] = [self.w[i].tolist() for i in range(self.nDoF)]
        
        m['params']={'nDoF':self.nDoF,'nPNs':self.nPNs,'nBasis':self.nBasis}
        m['params'].update({'k_NOI':self.k_NOI,'beta':self.beta,'delayTS':self.delayTS,'betaInertia':self.betaInertia,'SR':self.SR})

        
        m['constants']={}
        m['constants']['excFactR'] = [self.excFactR[i].tolist() for i in range(self.nDoF)]
        m['constants']['excFactD'] = [self.excFactD[i].tolist() for i in range(self.nDoF)]
        m['constants']['inhFactR'] = [self.inhFactR[i].tolist() for i in range(self.nDoF)]
        m['constants']['inhFactD'] = [self.inhFactD[i].tolist() for i in range(self.nDoF)]

        m['constants']['excThreshold'] = [self.excThreshold[i].tolist() for i in range(self.nDoF)]
        m['constants']['inhThreshold'] = [self.inhThreshold[i].tolist() for i in range(self.nDoF)]
        m['constants']['excScaleBasis'] = [self.excScaleBasis[i].tolist() for i in range(self.nDoF)]
        m['constants']['inhScaleBasis'] = [self.inhScaleBasis[i].tolist() for i in range(self.nDoF)]

       
        input_file = open(memoryFileName,'w')
        dump(m, input_file)
        input_file.close()

        print 'config saved'


    #########################################################
    def loadMemory(self, memoryFileName):
        
        f_config = open(memoryFileName,'r')
        m = loads(f_config.read())
        f_config.close()

        self.nDoF        = m['params']['nDoF']
        self.nPNs        = m['params']['nPNs']
        self.nBasis      = m['params']['nBasis']
        self.k_NOI       = m['params']['k_NOI']
        # self.beta        = m['params']['beta']
        self.delayTS     = m['params']['delayTS']
        self.betaInertia = m['params']['betaInertia']

        try:
        	self.SR = m['params']['SR']
        except:
        	pass

        self.w = [np.array(m['ws'][i]) for i in range(self.nDoF)]

        self.excFactR = [np.array(m['constants']['excFactR'][i]) for i in range(self.nDoF)]
        self.excFactD = [np.array(m['constants']['excFactD'][i]) for i in range(self.nDoF)]
        self.inhFactR = [np.array(m['constants']['inhFactR'][i]) for i in range(self.nDoF)]
        self.inhFactD = [np.array(m['constants']['inhFactD'][i]) for i in range(self.nDoF)]
        
        self.excThreshold = [np.array(m['constants']['excThreshold'][i]) for i in range(self.nDoF)]
        self.inhThreshold = [np.array(m['constants']['inhThreshold'][i]) for i in range(self.nDoF)]
        self.excScaleBasis = [np.array(m['constants']['excScaleBasis'][i]) for i in range(self.nDoF)]
        self.inhScaleBasis = [np.array(m['constants']['inhScaleBasis'][i]) for i in range(self.nDoF)]

        self.resetBasisState()

        print "memory loaded"



    ####################################################################
    # sets the in&out connections, learning rate, delay properties, etc.
    def plug(self, nDoF, nBasis, nPNs, k_NOI, beta, delay, SR):

        print "receiveplug"

        [self.SR, self.nPNs, self.nDoF, self.nBasis, self.k_NOI, self.beta, delay] = [SR, nPNs, nDoF, nBasis, k_NOI, beta, delay]

        self.delayTS = int(delay * self.SR)

        print "beta", beta
        print "SR", SR
        print "self  ", self

        self.initializeBasis()
        self.resetBasisState()

    ########################################################
    # reads a file with the cortical basis config
    def config(self, configFilename):

        print "reading config"

        f_cfg=open(configFilename,'r')
        self.cfg = loads(f_cfg.read())
        f_cfg.close()

    ########################################################
    def initTrial(self):
        
        for i in range(self.nDoF):
            self.excFirstInt[i] *= 0
            self.excSecInt[i]   *= 0
            self.inhFirstInt[i] *= 0
            self.inhSecInt[i]   *= 0
        
        self.pastActionsBuffer=np.zeros([self.delayTS,self.nDoF])

        # create new data field
        self.debug_data['trials'].append({})
        self.debug_data['trials'][-1]['steps']=[]

        if self.cTrial==0:
            if self.save_bases:
                self.debug_data['basis']={}
                self.debug_data['basis']['tot']=[]
                self.debug_data['basis']['exc']=[]
                self.debug_data['basis']['inh']=[]
            self.debug_data['inputs']=[]


    def update(self):
        self.doUpdate()


    def endTrial(self):
        
        if self.save_ws:
            self.debug_data['trials'][self.cTrial]['ws']       = [self.w[i].tolist() for i in range(self.nDoF)]
            self.debug_data['trials'][self.cTrial]['wChanges'] = [self.wChange[i].tolist() for i in range(self.nDoF)]

        self.doUpdate()

        self.cTrial += 1

    ########################################################
    def printDebug(self, debugFilename):
        
        debug_file = open(debugFilename,'w')

        dump(self.debug_data, debug_file)

        debug_file.close()

    ########################################################
    def freeze(self):
        self.beta = 0.0

        # stop momentum as well
        for i in range(self.nDoF):
            self.wChange[i] *= 0.0

        print 'frozen'

    ########################################################
    def input(self, dataArray):

        thisInput = np.array(dataArray[0:self.nPNs])
        target = dataArray[self.nPNs:(self.nPNs+self.nDoF)]
        
        self.error=np.array(target)-self.k_NOI*self.pastActionsBuffer[self.index][:];
            
        thisInputExpanded=np.tile(thisInput,[self.nBasis,1]).flatten(2) 

        for idof in range(self.nDoF):
            self.micro_circuit(idof, thisInputExpanded)
            
            self.actions[idof]=np.mean(self.out[idof])
            # self.actions[idof]=np.mean(self.out[idof])
            self.pastActionsBuffer[self.index][idof]=self.actions[idof]
        
        self.debug_data['trials'][-1]['steps'].append({})
        self.debug_data['trials'][-1]['steps'][-1]['error']=self.error.tolist()
        self.debug_data['trials'][-1]['steps'][-1]['target']=target
        self.debug_data['trials'][-1]['steps'][-1]['actions']=self.actions.tolist()
        self.debug_data['trials'][-1]['steps'][-1]['inputs']=thisInput.tolist()

        self.index = (self.index+1) % self.delayTS

        self.cCounter+=1

        return self.actions


    def setZeros(self):
        return  [np.zeros(self.nBasis*self.nPNs) for x in range(self.nDoF)]

    def resetBasisState(self):

        self.excFirstInt = self.setZeros()
        self.excSecInt   = self.setZeros()
        self.inhFirstInt = self.setZeros()
        self.inhSecInt   = self.setZeros()

        self.w       = self.setZeros()
        self.wChange = self.setZeros()
        self.act     = self.setZeros()

        self.out = self.setZeros()
        self.bufferOut =  [np.zeros([self.delayTS,self.nBasis*self.nPNs]) for x in range(self.nDoF)]
        
        self.index = 0

        self.actions=np.zeros(self.nDoF)


    # aux function for init Basis
    def randRange(self, interval):
        return interval[0]+(interval[1]-interval[0])*np.random.rand(self.nDoF,self.nBasis*self.nPNs)


    def initializeBasis(self):

        [excTauR, excTauD]=[np.exp(self.randRange(np.log(self.cfg['rangeR']))), \
            np.exp(self.randRange(np.log(self.cfg['rangeD'])))]
        
        excThresholdFactor=self.randRange(self.cfg['rangeExcThresholdFactor'])
        inhThresholdFactor=self.randRange(self.cfg['rangeInhThresholdFactor'])

        [excMax, inhMax] = [self.randRange(self.cfg['rangeExcMax']), \
            self.randRange(self.cfg['rangeInhMax'])]

        excFactR_ = np.exp(-1.0/(excTauR*self.SR))
        excFactD_ = np.exp(-1.0/(excTauD*self.SR))
        
        tmax=(excTauR*excTauD/(excTauD-excTauR))*(np.log(excTauD)-np.log(excTauR))
        maxVal=(np.exp(-tmax/excTauR)-np.exp(-tmax/excTauD))/((excTauR-excTauD)*self.SR)
        excThreshold_=excThresholdFactor*maxVal
        excScaleBasis_=excMax/(maxVal-excThreshold_)
        
        inhTauR=excTauR*self.randRange(self.cfg['inhibitionTimeFactorRange'])
        inhTauD=excTauD*self.randRange(self.cfg['inhibitionTimeFactorRange'])

        inhFactR_ = np.exp(-1.0/(inhTauR*self.SR))
        inhFactD_ = np.exp(-1.0/(inhTauD*self.SR))
        
        tmax=(inhTauR*inhTauD/(inhTauD-inhTauR))*(np.log(inhTauD)-np.log(inhTauR))
        maxVal=(np.exp(-tmax/inhTauR)-np.exp(-tmax/inhTauD))/((inhTauR-inhTauD)*self.SR)

        inhThreshold_=inhThresholdFactor*maxVal
        inhScaleBasis_=inhMax/(maxVal-inhThreshold_)
        
        self.excFactR = [excFactR_[:][i] for i in range(self.nDoF)]
        self.excFactD = [excFactD_[:][i] for i in range(self.nDoF)]
        self.inhFactR = [inhFactR_[:][i] for i in range(self.nDoF)]
        self.inhFactD = [inhFactD_[:][i] for i in range(self.nDoF)]
        
        self.excThreshold = [excThreshold_[:][i] for i in range(self.nDoF)]
        self.inhThreshold =  [inhThreshold_[:][i] for i in range(self.nDoF)]
        self.excScaleBasis =  [excScaleBasis_[:][i] for i in range(self.nDoF)]
        self.inhScaleBasis =  [inhScaleBasis_[:][i] for i in range(self.nDoF)]


    def __init__(self):

        # fake initializations
        [self.nDoF, self.nPNs, self.nBasis, self.k_NOI, self.beta, self.delayTS] = [0,0,0,0,0,0]
        self.cfg = 0
        [self.w, self.wChange, self.act, self.excFactR, self.excFactD, self.inhFactR, self.inhFactD] = [0,0,0,0,0,0,0]
        [self.excThreshold, self.inhThreshold, self.excScaleBasis, self.inhScaleBasis] = [0,0,0,0]
        [self.excFirstInt, self.excSecInt, self.inhFirstInt, self.inhSecInt] = [0,0,0,0]
        [self.out, self.bufferOut] = [0,0]
        [self.error, self.actions, self.pastActionsBuffer] = [0,0,0]
        
        self.index = 0
        self.save_bases = False
        self.save_ws = True 

        self.debug_data = {}

        self.cTrial = 0
        self.cStep = 0
        self.cCounter = 0

        self.lastUpdate = 0

        self.betaInertia = 0.9
        self.cTrial = 0 

        self.debug_data['trials']=[]
        if self.save_bases:
            self.debug_data['basis']=[]