# -*- coding: utf-8 -*-
"""
Created on Fri Aug 16 15:12:50 2019

@author: Thunderson-RJ
"""
import numpy as np

class LMS:
    def __init__(self, FilterOrder, InitialValue = 0):
        NoCoef = FilterOrder + 1
        if (InitialValue == 0):
            self.coef = np.zeros(NoCoef, dtype=complex)
        else:
            assert len(InitialValue) == NoCoef
            self.coef = np.array(InitialValue)
    
    #Updates the filter coefficients while storing previous iterations
    #(Mainly for plotting or academic purposes - heavy memory usage)
    def fit(self, DesiredOutput, Input, Mu):
        assert len(DesiredOutput) == len(Input)
        NoIterations = len(Input)
        NoCoef       = len(self.coef)
        
        #Converting to Numpy arrays
        DesiredOutput = np.array(DesiredOutput)
        Input         = np.array(Input)
        
        #Pre Allocations
        ErrorVector  = np.zeros(NoIterations, dtype = complex);
        OutputVector = np.zeros(NoIterations, dtype = complex);
        CoefVector   = np.zeros([(NoIterations+1), NoCoef], dtype = complex);
        
        #Input initial conditions (assumed relaxed)
        InputVector   = np.zeros(NoCoef, dtype = complex)
        CoefVector[0] = self.coef
        
        #Filter training
        for it in range(NoIterations):
            InputVector    = np.roll(InputVector,1)
            InputVector[0] = Input[it]
            
            OutputVector[it] = np.vdot(CoefVector[it],InputVector)
            ErrorVector[it]  = DesiredOutput[it] - OutputVector[it]
            CoefVector[it+1] = CoefVector[it] + np.conj(ErrorVector[it])*Mu*InputVector
        
        self.coef = CoefVector[-1]
        return OutputVector, ErrorVector, CoefVector