#%load_ext autoreload
#%autoreload 2
import yfinance as yf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats

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
    
    def AnnualizingReturn(self,PerReturn,PerInAYear):
        '''
        Funcion que calcula el retorno anualizado
        '''
        ret = (1+PerReturn)**PerInAYear - 1
        return ret
    
    def MeanReturn(self,Base):
        Base = Base.mean()
        return Base
    
    def VolReturn(self,Base):
        Base = Base.std()
        return Base
    
    def Sesgo(self,Base):
        DesMean = Base-Base.mean()
        sigma_Base = Base.std(ddof=0)
        exp = (DesMean**3).mean()
        return exp/sigma_Base**3
    
    def kurtosis(self,Base):
        """
        Alternative to scipy.stats.kurtosis()
        Computes the kurtosis of the supplied Series or DataFrame
        Returns a float or a Series
        """
        demeaned_Base = Base - Base.mean()
        # use the population standard deviation, so set dof=0
        sigma_Base = Base.std(ddof=0)
        exp = (demeaned_Base**4).mean()
        return exp/sigma_Base**4
    
    def drawdown(self,return_series: pd.Series):
        """Takes a time series of asset returns.
           returns a DataFrame with columns for
           the wealth index, 
           the previous peaks, and 
           the percentage drawdown
        """
        wealth_index = 1000*(1+return_series).cumprod()
        previous_peaks = wealth_index.cummax()
        drawdowns = (wealth_index - previous_peaks)/previous_peaks
        return pd.DataFrame({"Wealth": wealth_index, 
                             "Previous Peak": previous_peaks, 
                             "Drawdown": drawdowns})
    
    def AnnualizedVol(self,Base,Per):
        Base = self.VolReturn(Base)*np.sqrt(Per)
        return Base
    
    def AnnualizedMean(self,Base,DaysBase):
        Days = Base.shape[0]
        Base = (Base+1).prod()**(DaysBase/Days) - 1
        return Base
    
    def is_normal(self,r,level=0.01):
        """
        Applies the Jarque-Bera test to determine if a Series is normal or not
        Test is applied at the 1% level by default
        Returns True if the hypothesis of normality is accepted, False otherwise
        """
        if isinstance(r, pd.DataFrame):
            return r.aggregate(self.is_normal)
        else:
            statistic, p_value = scipy.stats.jarque_bera(r)
            return p_value > level
        

        
     
        
    




