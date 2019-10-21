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
            
            
    def GetDataFromYahooSeveral(self,start,end,period,columna,*args):
        '''
         Junta los Precios de varios Tickers
        '''
        Base = pd.DataFrame()
        for i in args:
            data = self.GetDataFromYahoo(i,start,end,period)[columna]
            Base = pd.merge(Base,data,how='outer',left_index=True,right_index=True)
            Base = Base.fillna(method='ffill')
        Base.columns = args
            
            
        return Base
            
class ReturnOps():
    def ComputeOnereturn(self,Base,columns):
        try:
            returns = Base[columns].pct_change()
            returns = returns.dropna()
            return returns
        except:
            print('Error no existe la columna o el archivo esta en el formato incorrecto')
    def ComputeSeveralReturns(self,Base):
        '''
        Funcion que toma un dataframe de precios y calcula los retornos
        '''
        returns = Base.pct_change()
        returns = returns.dropna()
        return returns
        
            
            
            
        return None
    def AnnualizingReturn(self,PerReturn,PerInAYear):
        '''
        Funcion que calcula el retorno anualizado
        '''
        ret = (1+PerReturn)**PerInAYear - 1
        return ret
        
     
        
    




