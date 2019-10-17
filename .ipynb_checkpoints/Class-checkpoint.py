#%load_ext autoreload
#%autoreload 2
import yfinance as yf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

class GetAndFormatTheData:
    #def __init__(self,Ticker):
    #    self.Ticker = Ticker
    
    def GetDataFromYahoo(self,Ticker,start,end,period):
        try:
            data = yf.download(Ticker,start=start,end=end,period=period)
            return data
        except:
            print('Error El Ticker no existe')
            
class ReturnOps:
    def ComputeOnereturn(self,Base,columns):
        try:
            returns = Base[columns].pct_change()
            returns = returns.dropna()
            return returns
        except:
            print('Error no existe la columna o el archivo esta en el formato incorrecto')
    def ComputeSeveralReturns(*kwargs):
        return returns
    def AnnualizingReturn(self,Percentage,PerInAYear):
        ret = (1+Percentage)**PerInAYear - 1
        return ret
        
     
        
    




