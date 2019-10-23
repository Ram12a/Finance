#%load_ext autoreload
#%autoreload 2
import yfinance as yf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats
from scipy.stats import norm

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
    def GetTheMinimunDateDraadown(self,Serie):
        Serie = self.drawdown(Serie)["Drawdown"].idxmin()
        return Serie
    
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
    
    def semideviation(self,r):
        """
        Returns the semideviation aka negative semideviation of r
        r must be a Series or a DataFrame
        """
        is_negative = r < 0
        return r[is_negative].std(ddof=0)
    
    def varhistoric(self,r, level=5):
        """
        Returns the historic Value at Risk at a specified level
        i.e. returns the number such that "level" percent of the returns
        fall below that number, and the (100-level) percent are above
        """
        if isinstance(r, pd.DataFrame):
            return r.aggregate(self.varhistoric, level=level)

        elif isinstance(r, pd.Series):
            return -np.percentile(r, level)
        else:
            raise TypeError("Expected r to be a Series or DataFrame")
    
    def cvarhistoric(self,r, level=5):
        """
        Computes the Conditional VaR of Series or DataFrame
        """
        if isinstance(r, pd.Series):
            is_beyond = r <= -self.varhistoric(r, level=level)
            return -r[is_beyond].mean()
        elif isinstance(r, pd.DataFrame):
            return r.aggregate(self.cvarhistoric, level=level)
        else:
            raise TypeError("Expected r to be a Series or DataFrame")
    
    def vargaussian(self,r, level=5, modified=False):
        """
        Returns the Parametric Gauusian VaR of a Series or DataFrame
        If "modified" is True, then the modified VaR is returned,
        using the Cornish-Fisher modification
        """
        # compute the Z score assuming it was Gaussian
        z = norm.ppf(level/100)
        if modified:
            # modify the Z score based on observed skewness and kurtosis
            s = self.Sesgo(r)
            k = self.kurtosis(r)
            z = (z +
                    (z**2 - 1)*s/6 +
                    (z**3 -3*z)*(k-3)/24 -
                    (2*z**3 - 5*z)*(s**2)/36
                )
        return -(r.mean() + z*r.std(ddof=0))

class Portfolio(ReturnOps):
    def sharpe_ratioAnnual(self,Base,riskfree_rate,periods_per_year):
        """
        Computes the annualized sharpe ratio of a set of returns
        """
        # convert the annual riskfree rate to per period
        rf_per_period = (1+riskfree_rate)**(1/periods_per_year)-1
        excess_ret = Base - rf_per_period
        ann_ex_ret = self.AnnualizedMean(excess_ret,periods_per_year)
        ann_vol = self.AnnualizedVol(Base,periods_per_year)
        return ann_ex_ret/ann_vol
    
    
    def portfolio_return(weights, returns):
        """
        Computes the return on a portfolio from constituent returns and weights
        weights are a numpy array or Nx1 matrix and returns are a numpy array or Nx1 matrix
        """
        return weights.T @ returns
    
    def portfolio_vol(weights, covmat):
        """
        Computes the vol of a portfolio from a covariance matrix and constituent weights
        weights are a numpy array or N x 1 maxtrix and covmat is an N x N matrix
        """
        return (weights.T @ covmat @ weights)**0.5
    def plot_ef2(n_points, er, cov):
        """
        Plots the 2-asset efficient frontier
        """
        if er.shape[0] != 2 or er.shape[0] != 2:
            raise ValueError("plot_ef2 can only plot 2-asset frontiers")
        weights = [np.array([w, 1-w]) for w in np.linspace(0, 1, n_points)]
        rets = [portfolio_return(w, er) for w in weights]
        vols = [portfolio_vol(w, cov) for w in weights]
        ef = pd.DataFrame({
            "Returns": rets, 
            "Volatility": vols
        })
        return ef.plot.line(x="Volatility", y="Returns", style=".-")



    
    




        
    




