from perforatedai import pb_globals as PBG
from safetensors.torch import load_file
import copy

import torch.nn as nn
import torch
import pdb

from threading import Thread


doingThreading = False
loadedFullPrint = False


def convertModule(net,  depth, nameSoFar):
    if(type(net) is PAIModulePyThread):
        print('Something in your model is pointed to twice by two different variables. Skipping second instance')
        print(net)
        return net
    allMembers = net.__dir__()
    # print(f"All members is in this network {allMembers}")
    # If this module is a Module List or Sequential go through each module
    if issubclass(type(net),nn.Sequential) or issubclass(type(net),nn.ModuleList):
        
        submoduleID = 0
        seqLen = len(net)
        while submoduleID < seqLen:
            # If it has a substitution in modulesToReplace make that substitution
            if type(net[submoduleID]) in PBG.modulesToReplace:
                
                net[submoduleID] = replacePredefinedModules(net[submoduleID],  getPretrainedPBVar(submoduleID))
            # If it is set as a module to convert make the converstion
            if (type(net[submoduleID]) in PBG.modulesToConvert
                or
                type(net[submoduleID]).__name__ in PBG.moduleNamesToConvert):
                net[submoduleID] = PAIModulePyThread(net[submoduleID], nameSoFar + '.' + str(submoduleID))
               
            # Otherwise check the module recursively if there are other modules to convert
            else:
                if(net != net[submoduleID]):
                    net[submoduleID] = convertModule(net[submoduleID],  depth + 1, nameSoFar + '.' + str(submoduleID))
                               
            submoduleID += 1
    # If the module is listed in ones to skip just continue
    elif(type(net) in PBG.modulestoSkip):
        return net
    # If it is neither a sequential nor a skipped module must check conversion for each member variable
    else:
        for member in allMembers:    
            # print("")
            # print(member)    
            # If it has a substitution in modulesToReplace make that substitution
            if type(getattr(net,member,None)) in PBG.modulesToReplace:
               
                setattr(net,member,replacePredefinedModules(getattr(net,member,None)))
            # If it is set as a module to convert make the converstion
            if (type(getattr(net,member,None)) in PBG.modulesToConvert
                or
                type(getattr(net,member,None)).__name__ in PBG.moduleNamesToConvert):
                
                setattr(net,member,PAIModulePyThread(getattr(net,member),nameSoFar + '.' + member))
            # Otherwise, if it is a module, check the module recursively if there are other modules to convert
            elif issubclass(type(getattr(net,member,None)),nn.Module):
                if(net != getattr(net,member)):
                    setattr(net,member,convertModule(getattr(net,member), depth+1, nameSoFar + '.' + member))
                    
    return net


def convertNetwork(net, layerName=''):
    # If the net itself has a substitution make that substitution first
    if type(net) in PBG.modulesToReplace:
        net = replacePredefinedModules(net)
    # If the net itself should be converted make the converstion
    if(type(net) in PBG.modulesToConvert):
        if(layerName == ''):
            print('converting a single layer without a name, add a layerName param to the call')
            exit(-1)
        
        net = PAIModulePyThread(net, layerName)
    # Otherwise, check the module recursively if there are other modules to convert
    else:
        print('starting main call')

        net = convertModule(net,  0, 'model')
    return net

def getPAIModules(net, depth):
    allMembers = net.__dir__()
    thisList = []
    if issubclass(type(net),nn.Sequential) or issubclass(type(net),nn.ModuleList):
        for submoduleID in range(len(net)):
            if net[submoduleID] is net:
                continue
            if type(net[submoduleID]) is PAIModulePyThread:
                thisList = thisList + [net[submoduleID]]
            else:
                thisList = thisList + getPAIModules(net[submoduleID], depth + 1)            
    else:
        for member in allMembers:        
            if getattr(net,member,None) is net:
                continue
            if type(getattr(net,member,None)) is PAIModulePyThread:
                thisList = thisList + [getattr(net,member)]
            elif issubclass(type(getattr(net,member,None)),nn.Module):
                thisList = thisList + getPAIModules(getattr(net,member), depth+1)
    return thisList 

def loadPAIModel(net, filename):
    net = convertNetwork(net)
    
    stateDict = load_file(filename)
    
    pbModules = getPAIModules(net,0)

    # print(net)
    if(pbModules == []):
        print('No PAI modules were found something went wrong with convert network')
        # sys.exit()
    
    for module in pbModules:
        
        moduleName = module.name
       
        if moduleName[:5] == 'model':
            #strip "model."
            moduleName = moduleName[6:]
        # If it was a dataparallel also remove 'module' from the name
        if moduleName[:6] == 'module':
            #strip the "module."
            moduleName = moduleName[7:]        
        # Then instantiate as many Dendrites as were created during training
        numCycles = int(stateDict[moduleName + '.numCycles'].item())
       
        nodeCount = 10
        #also extract view tuple
        
        if(numCycles > 0):
            module.simulateCycles(numCycles, nodeCount)
            
        # module.register_buffer('skipWeights', torch.zeros(stateDict[moduleName + '.skipWeights'].shape))
        module.register_buffer('skipWeights', stateDict[moduleName + '.skipWeights'])
        module.register_buffer('moduleID', stateDict[moduleName + '.moduleID'])
        module.register_buffer('viewTuple', stateDict[moduleName + '.viewTuple'])
    
    # torch.save(net,"trainedFinalBufferIncluded.pt")
    # net.load_state_dict(stateDict)
    
    return net
    #figure out if doing this 'thread' stuff is actually helping at all.
    #If its not just get rid of it to simplify things.
    #to test this will have to first get loadPAIModel actually set up and working then run a test with and #without threading.



class PAIModulePyThread(nn.Module):
    def __init__(self, originalModule, name):
        super(PAIModulePyThread, self).__init__()
        self.name = name
        self.register_buffer('nodeIndex', torch.tensor(-1))
        self.register_buffer('moduleID', torch.tensor(-1))
        self.register_buffer('numCycles', torch.tensor(-1))
        self.register_buffer('viewTuple', torch.tensor(-1))
        self.processorArray = []
        self.processor = None
        self.layerArray = nn.ModuleList([originalModule])
        self.layerArray[-1].register_buffer('moduleID', torch.tensor(-1))

        # If this original module has processing functions save the processor
        if(type(originalModule) in PBG.modluesWithProcessing):
            moduleIndex = PBG.modluesWithProcessing.index(type(originalModule))
            
            self.processor = PBG.moduleProcessingClasses[moduleIndex]()
        elif(type(originalModule).__name__ in PBG.moduleNamesWithProcessing):
            moduleIndex = PBG.moduleNamesWithProcessing.index(type(originalModule).__name__)
            self.processor = PBG.moduleByNameProcessingClasses[moduleIndex]()
           
        self.register_buffer('moduleID', torch.tensor(0))

    def simulateCycles(self, numCycles, nodeCount):
        for i in range(0,numCycles,2):
            self.layerArray.append(copy.deepcopy(self.layerArray[0]))
            self.layerArray[-1].register_buffer('moduleID', torch.tensor(-1))
            if(not self.processor is None):
                processor = copy.deepcopy(self.processor)
                processor.pre=self.processor.pre_d
                processor.post=self.processor.post_d
                self.processorArray.append(processor)
            else:
                self.processorArray.append(None)

        if(not self.processor is None):
            processor = copy.deepcopy(self.processor)
            processor.pre=self.processor.post_n1
            processor.post=self.processor.post_n2
            self.processorArray.append(processor)
        else:
            self.processorArray.append(None)

    def processAndForward(self, *args2, **kwargs2):
        c = args2[0]
        pbOuts = args2[1]
        args2 = args2[2:]
        if(self.processorArray[c] != None):
            outValues = self.processorArray[c].pre(*args2, **kwargs2)
        outValues = self.layerArray[c](*args2, **kwargs2)
        if(self.processorArray[c] != None):
            out = self.processorArray[c].post(outValues)
        else:
            out = outValues
        pbOuts[c] = out
    
    def processAndPre(self, *args, **kwargs):
        pbOuts = args[0]
        args = args[1:]
        out = self.layerArray[-1].forward(*args, **kwargs)
        if not self.processorArray[-1] is None:
            out = self.processorArray[-1].pre(out)
        pbOuts[len(self.layerArray)-1] = out
        
    def forward(self, *args, **kwargs):
        #this is currently false anyway, just remove the doing multi idea
        doingMulti = doingThreading
        pbOuts = [None] * len(self.layerArray)
        threads = {}
        for c in range(0,len(self.layerArray)-1):
            args2, kwargs2 = args, kwargs
            if(doingMulti):
                threads[c] = Thread(target=self.processAndForward, args=(c, pbOuts, *args), kwargs=kwargs)
            else:
                self.processAndForward(c, pbOuts, *args2, **kwargs2)
        if(doingMulti):
            threads[len(self.layerArray)-1] = Thread(target=self.processAndPre, args=(pbOuts, *args), kwargs=kwargs)
        else:
            self.processAndPre(pbOuts, *args, **kwargs)
        if(doingMulti):
            for i in range(len(pbOuts)):
                threads[i].start()
            for i in range(len(pbOuts)):
                threads[i].join()
        for outIndex in range(0,len(self.layerArray)):
            currentOut = pbOuts[outIndex]
            ## Can I generate this view tuple easily to be saved and loaded instead of generated?
            #if(self.viewTuple == []):
                #for dim in range(len(currentOut.shape)):
                    #if dim == self.nodeIndex:
                        #self.viewTuple.append(-1)
                        #continue
                    #self.viewTuple.append(1)
            if(len(self.layerArray) > 1):
                for inIndex in range(0,outIndex):
                    skip_weight = self.skipWeights[outIndex][inIndex, :].view(self.viewTuple.tolist()).squeeze(0)
                    pbOut = pbOuts[inIndex]

                    currentOut += skip_weight.to(currentOut.device) * pbOut

                    # currentOut += self.skipWeights[outIndex][inIndex,:].view(self.viewTuple.tolist()).to(currentOut.device) * pbOuts[inIndex]    
                if(outIndex < len(self.layerArray)-1):
                    currentOut = PBG.PBForwardFunction(currentOut)
            pbOuts[outIndex] = currentOut
        if not self.processorArray[-1] is None:
            currentOut = self.processorArray[-1].post(currentOut)
        return currentOut

