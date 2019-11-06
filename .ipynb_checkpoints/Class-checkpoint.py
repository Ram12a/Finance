#%load_ext autoreload
#%autoreload 2
import yfinance as yf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats
from scipy.stats import norm
from scipy.optimize import minimize

class GetAndFormatTheData:
    def GetDataFromYahoo(self,Ticker,start,end,interval):
        '''
          The function get the data from Yahoo finance
    
          Inputs: Ticker:string,date:string('YYYY-MM-DD'),
    
          period:string(1d,5d,1wk,1mo,3mo)
    
          output:return a data frame OHLS
        '''
        try:
            data = yf.download(Ticker,start=start,end=end,interval=interval)
            return data
        except:
            print('Error El Ticker no existe')
            
            
    def GetDataFromYahooSeveral(self,start,end,interval,columna,*args):
        '''
         The function get the data from Yahoo finance,join the columns and fill the values with the last value.
    
         Inputs: Ticker:string,date:string('YYYY-MM-DD'),period:string(1d,5d,1wk,1mo,3mo),columna:string.(Open,High,Low,Close,Adj Close,Volume),args: Several tickers
    
         output:return a data frame with the selected column and the tickers
        '''
        Base = pd.DataFrame()
        for i in args:
            assert isinstance(i,str),'The value is not string'
            try:
                data = self.GetDataFromYahoo(i,start,end,interval)[columna]
                Base = pd.merge(Base,data,how='outer',left_index=True,right_index=True)
                Base = Base.fillna(method='ffill')
            except:
                print("El Ticker no existe o no exsten datos en el Ticker %s"%(i))
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
    def compound(r):
        """
        returns the result of compounding the set of returns in r
        """
        return np.expm1(np.log1p(r).sum())
    
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
        '''
         Per the number of period in a year
        '''
        Base = self.VolReturn(Base)*np.sqrt(Per)
        return Base
    
    def AnnualizedMean(self,Base,DaysBase):
        '''
        The values should be divided by 100
        '''
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
    
    def varhistoric(self,r,level=5):
        """
        Returns the historic Value at Risk at a specified level
        i.e. returns the number such that "level" percent of the returns
        fall below that number, and the (100-level) percent are above
        """
        if isinstance(r,pd.DataFrame):
            return r.aggregate(self.varhistoric, level=level)

        elif isinstance(r, pd.Series):
            return -np.percentile(r, level)
        else:
            raise TypeError("Expected r to be a Series or DataFrame")
    
    def cvarhistoric(self,r,level=5):
        """
        Computes the Conditional VaR of Series or DataFrame
        """
        if isinstance(r,pd.Series):
            is_beyond = r <= -self.varhistoric(r, level=level)
            return -r[is_beyond].mean()
        elif isinstance(r,pd.DataFrame):
            return r.aggregate(self.cvarhistoric, level=level)
        else:
            raise TypeError("Expected r to be a Series or DataFrame")
    
    def vargaussian(self,r,level=5,modified=False):
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
    
    def annualize_rets(self,r, periods_per_year):
        """
        Annualizes a set of returns
        We should infer the periods per year
        but that is currently left as an exercise
        to the reader :-)
        """
        compounded_growth = (1+r).prod()
        n_periods = r.shape[0]
        return compounded_growth**(periods_per_year/n_periods)-1


    def annualize_vol(self,r,periods_per_year):
        """
        Annualizes the vol of a set of returns
        We should infer the periods per year
        but that is currently left as an exercise
        to the reader :-)
        """
        return r.std()*(periods_per_year**0.5)
    
    def RollingMeanAnnualizeReturn(self,BaseRet,windows,periodPerYear):
        tmi_tr36rets = BaseRet.rolling(window=windows).aggregate(self.annualize_rets, periods_per_year=periodPerYear)
        return tmi_tr36rets
    
    def RollingMeanCorr(self,BaseRet,windows):
        BaseRetRetPort.index.name = 'date'
        ts_corr = BaseRet.rolling(window=windows).corr()
        ts_corr.index.name = 'date'
        print(ts_corr.tail())
        ind_tr36corr = ts_corr.groupby(level='date').apply(lambda cormat: cormat.values.mean())
        return ind_tr36corr
        

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
    
    
    def portfolio_return(self,weights,returns):
        """
        Computes the return on a portfolio from constituent returns and weights
        weights are a numpy array or Nx1 matrix and returns are a numpy array or Nx1 matrix
        """
        return weights.T @ returns
    
    def portfolio_vol(self,weights,covmat):
        """
        Computes the vol of a portfolio from a covariance matrix and constituent weights
        weights are a numpy array or N x 1 maxtrix and covmat is an N x N matrix
        """
        return (weights.T @ covmat @ weights)**0.5
    
    def plot_ef2(self,n_points,er,cov):
        """
        Plots the 2-asset efficient frontier
        """
        if er.shape[0] != 2 or er.shape[0] != 2:
            raise ValueError("plot_ef2 can only plot 2-asset frontiers")
        weights = [np.array([w, 1-w]) for w in np.linspace(0, 1, n_points)]
        rets = [self.portfolio_return(w, er) for w in weights]
        vols = [self.portfolio_vol(w, cov) for w in weights]
        ef = pd.DataFrame({
            "Returns": rets, 
            "Volatility": vols
        })
        return ef.plot.line(x="Volatility", y="Returns", style=".-")

    
    def minimize_vol(self,target_return,er,cov):
        """
        Returns the optimal weights that achieve the target return
        given a set of expected returns and a covariance matrix
        """
        n = er.shape[0]
        init_guess = np.repeat(1/n, n)
        bounds = ((0.0, 1.0),) * n # an N-tuple of 2-tuples!
        # construct the constraints
        weights_sum_to_1 = {'type': 'eq',
                            'fun': lambda weights: np.sum(weights) - 1
        }
        return_is_target = {'type': 'eq',
                            'args': (er,),
                            'fun': lambda weights, er: target_return - self.portfolio_return(weights,er)
        }
        weights = minimize(self.portfolio_vol, init_guess,
                           args=(cov,), method='SLSQP',
                           options={'disp': False},
                           constraints=(weights_sum_to_1,return_is_target),
                           bounds=bounds)
        return weights.x
    
    def optimal_weights(self,n_points,er,cov):
        """
        Returns a list of weights that represent a grid of n_points on the efficient frontier
        """
        target_rs = np.linspace(er.min(), er.max(),n_points)
        weights = [self.minimize_vol(target_return,er,cov) for target_return in target_rs]
        return weights
    
    
    def msr(self,riskfree_rate,er,cov):
        """
        Returns the weights of the portfolio that gives you the maximum sharpe ratio
        given the riskfree rate and expected returns and a covariance matrix
        """
        n = er.shape[0]
        init_guess = np.repeat(1/n, n)
        bounds = ((0.0, 1.0),) * n # an N-tuple of 2-tuples!
        # construct the constraints
        weights_sum_to_1 = {'type': 'eq',
                            'fun': lambda weights: np.sum(weights) - 1
        }
        def neg_sharpe(weights,riskfree_rate,er,cov):
            
            """
            Returns the negative of the sharpe ratio
            of the given portfolio
            """
            
            r = self.portfolio_return(weights,er)
            vol = self.portfolio_vol(weights,cov)
            return -(r - riskfree_rate)/vol

        weights = minimize(neg_sharpe,init_guess,
                           args=(riskfree_rate, er, cov), method='SLSQP',
                           options={'disp': False},
                           constraints=(weights_sum_to_1,),
                           bounds=bounds)
        return weights.x
    
    def gmv(self,cov):
        """
        Returns the weights of the Global Minimum Volatility portfolio
        given a covariance matrix
        """
        n = cov.shape[0]
        return self.msr(0,np.repeat(1,n),cov)
    
    def plot_ef(self,n_points,er,cov,style='.-',legend=False,show_cml=False,riskfree_rate=0,show_ew=False,show_gmv=False):
        """
        Plots the multi-asset efficient frontier
        """
        weights = self.optimal_weights(n_points,er,cov)
        rets = [self.portfolio_return(w, er) for w in weights]
        vols = [self.portfolio_vol(w, cov) for w in weights]
        ef = pd.DataFrame({
            "Returns": rets, 
            "Volatility": vols
        })
        ax = ef.plot.line(x="Volatility", y="Returns",style=style,legend=legend)
        if show_cml:
            ax.set_xlim(left = 0)
            # get MSR
            w_msr = self.msr(riskfree_rate,er,cov)
            r_msr = self.portfolio_return(w_msr,er)
            vol_msr = self.portfolio_vol(w_msr,cov)
            # add CML
            cml_x = [0, vol_msr]
            cml_y = [riskfree_rate,r_msr]
            ax.plot(cml_x, cml_y,color='green',marker='o',linestyle='dashed',linewidth=2,markersize=10)
        if show_ew:
            n = er.shape[0]
            w_ew = np.repeat(1/n, n)
            r_ew = self.portfolio_return(w_ew,er)
            vol_ew = self.portfolio_vol(w_ew,cov)
            # add EW
            ax.plot([vol_ew], [r_ew],color='goldenrod',marker='o',markersize=10)
        if show_gmv:
            w_gmv = self.gmv(cov)
            r_gmv = self.portfolio_return(w_gmv,er)
            vol_gmv = self.portfolio_vol(w_gmv,cov)
            # add EW
            ax.plot([vol_gmv],[r_gmv],color='midnightblue',marker='o',markersize=10)

            return ax

class CPPI(Portfolio):
    def run_cppi(self,risky_r, safe_r=None, m=3, start=1000, floor=0.8, riskfree_rate=0.03, drawdown=None):
        """
        Run a backtest of the CPPI strategy, given a set of returns for the risky asset
        Returns a dictionary containing: Asset Value History, Risk Budget History, Risky Weight History
        """
        # set up the CPPI parameters
        dates = risky_r.index
        n_steps = len(dates)
        account_value = start
        floor_value = start*floor
        peak = account_value
        if isinstance(risky_r, pd.Series): 
            risky_r = pd.DataFrame(risky_r, columns=["R"])

        if safe_r is None:
            safe_r = pd.DataFrame().reindex_like(risky_r)
            safe_r.values[:] = riskfree_rate/12 # fast way to set all values to a number
        # set up some DataFrames for saving intermediate values
        account_history = pd.DataFrame().reindex_like(risky_r)
        risky_w_history = pd.DataFrame().reindex_like(risky_r)
        cushion_history = pd.DataFrame().reindex_like(risky_r)
        floorval_history = pd.DataFrame().reindex_like(risky_r)
        peak_history = pd.DataFrame().reindex_like(risky_r)

        for step in range(n_steps):
            if drawdown is not None:
                peak = np.maximum(peak, account_value)
                floor_value = peak*(1-drawdown)
            cushion = (account_value - floor_value)/account_value
            risky_w = m*cushion
            risky_w = np.minimum(risky_w, 1)
            risky_w = np.maximum(risky_w, 0)
            safe_w = 1-risky_w
            risky_alloc = account_value*risky_w
            safe_alloc = account_value*safe_w
            # recompute the new account value at the end of this step
            account_value = risky_alloc*(1+risky_r.iloc[step]) + safe_alloc*(1+safe_r.iloc[step])
            # save the histories for analysis and plotting
            cushion_history.iloc[step] = cushion
            risky_w_history.iloc[step] = risky_w
            account_history.iloc[step] = account_value
            floorval_history.iloc[step] = floor_value
            peak_history.iloc[step] = peak
        risky_wealth = start*(1+risky_r).cumprod()
        backtest_result = {
            "Wealth": account_history,
            "Risky Wealth": risky_wealth, 
            "Risk Budget": cushion_history,
            "Risky Allocation": risky_w_history,
            "m": m,
            "start": start,
            "floor": floor,
            "risky_r":risky_r,
            "safe_r": safe_r,
            "drawdown": drawdown,
            "peak": peak_history,
            "floor": floorval_history
        }
        return backtest_result
    
    def summary_stats(self,r,riskfree_rate=0.03):
        """
        Return a DataFrame that contains aggregated summary stats for the returns in the columns of r
        """
        ann_r = r.aggregate(self.annualize_rets, periods_per_year=12)
        ann_vol = r.aggregate(self.annualize_vol, periods_per_year=12)
        ann_sr = r.aggregate(self.sharpe_ratioAnnual, riskfree_rate=riskfree_rate, periods_per_year=12)
        dd = r.aggregate(lambda r: self.drawdown(r).Drawdown.min())
        skew = r.aggregate(self.Sesgo)
        kurt = r.aggregate(self.kurtosis)
        cf_var5 = r.aggregate(self.vargaussian, modified=True)
        hist_cvar5 = r.aggregate(self.cvarhistoric)
        return pd.DataFrame({
            "Annualized Return": ann_r,
            "Annualized Vol": ann_vol,
            "Skewness": skew,
            "Kurtosis": kurt,
            "Cornish-Fisher VaR (5%)": cf_var5,
            "Historic CVaR (5%)": hist_cvar5,
            "Sharpe Ratio": ann_sr,
            "Max Drawdown": dd
        })
    
class MontecarloPrices(CPPI):
    def gbm(self,n_years = 10, n_scenarios=1000, mu=0.07, sigma=0.15, steps_per_year=12, s_0=100.0, prices=True):
    
        """
        Evolution of Geometric Brownian Motion trajectories, such as for Stock Prices through Monte Carlo
        :param n_years:  The number of years to generate data for
        :param n_paths: The number of scenarios/trajectories
        :param mu: Annualized Drift, e.g. Market Return
        :param sigma: Annualized Volatility
        :param steps_per_year: granularity of the simulation
        :param s_0: initial value
        :return: a numpy array of n_paths columns and n_years*steps_per_year rows
        """
        # Derive per-step Model Parameters from User Specifications
        dt = 1/steps_per_year
        n_steps = int(n_years*steps_per_year) + 1
        # the standard way ...
        # rets_plus_1 = np.random.normal(loc=mu*dt+1, scale=sigma*np.sqrt(dt), size=(n_steps, n_scenarios))
        # without discretization error ...
        rets_plus_1 = np.random.normal(loc=(1+mu)**dt, scale=(sigma*np.sqrt(dt)), size=(n_steps, n_scenarios))
        rets_plus_1[0] = 1
        ret_val = s_0*pd.DataFrame(rets_plus_1).cumprod() if prices else rets_plus_1-1
        return ret_val
    
    def show_gbm(self,n_scenarios, mu, sigma):
        """
        Draw the results of a stock price evolution under a Geometric Brownian Motion model
        """
        s_0=100
        prices = self.gbm(n_scenarios=n_scenarios, mu=mu, sigma=sigma, s_0=s_0)
        ax = prices.plot(legend=False, color="indianred", alpha = 0.5, linewidth=2, figsize=(12,5))
        ax.axhline(y=100, ls=":", color="black")
        # draw a dot at the origin
        ax.plot(0,s_0, marker='o',color='darkred', alpha=0.2)
    
    def show_cppi(self,n_scenarios=50, mu=0.07, sigma=0.15, m=3, floor=0., riskfree_rate=0.03, steps_per_year=12, y_max=100):
        """
        Plot the results of a Monte Carlo Simulation of CPPI
        """
        start = 1000
        sim_rets = self.gbm(n_scenarios=n_scenarios, mu=mu, sigma=sigma, prices=False, steps_per_year=steps_per_year)
        risky_r = pd.DataFrame(sim_rets)
        # run the "back"-test
        btr = self.run_cppi(risky_r=pd.DataFrame(risky_r),riskfree_rate=riskfree_rate,m=m, start=start, floor=floor)
        wealth = btr["Wealth"]

        # calculate terminal wealth stats
        y_max=wealth.values.max()*y_max/100
        terminal_wealth = wealth.iloc[-1]

        tw_mean = terminal_wealth.mean()
        tw_median = terminal_wealth.median()
        failure_mask = np.less(terminal_wealth, start*floor)
        n_failures = failure_mask.sum()
        p_fail = n_failures/n_scenarios

        e_shortfall = np.dot(terminal_wealth-start*floor, failure_mask)/n_failures if n_failures > 0 else 0.0

        # Plot!
        fig, (wealth_ax, hist_ax) = plt.subplots(nrows=1, ncols=2, sharey=True, gridspec_kw={'width_ratios':[3,2]}, figsize=(24, 9))
        plt.subplots_adjust(wspace=0.0)

        wealth.plot(ax=wealth_ax, legend=False, alpha=0.3, color="indianred")
        wealth_ax.axhline(y=start, ls=":", color="black")
        wealth_ax.axhline(y=start*floor, ls="--", color="red")
        wealth_ax.set_ylim(top=y_max)

        terminal_wealth.plot.hist(ax=hist_ax, bins=50, ec='w', fc='indianred', orientation='horizontal')
        hist_ax.axhline(y=start, ls=":", color="black")
        hist_ax.axhline(y=tw_mean, ls=":", color="blue")
        hist_ax.axhline(y=tw_median, ls=":", color="purple")
        hist_ax.annotate(f"Mean: ${int(tw_mean)}", xy=(.7, .9),xycoords='axes fraction', fontsize=24)
        hist_ax.annotate(f"Median: ${int(tw_median)}", xy=(.7, .85),xycoords='axes fraction', fontsize=24)
        if (floor > 0.01):
            hist_ax.axhline(y=start*floor, ls="--", color="red", linewidth=3)
            hist_ax.annotate(f"Violations: {n_failures} ({p_fail*100:2.2f}%)\nE(shortfall)=${e_shortfall:2.2f}", xy=(.7, .7), xycoords='axes fraction', fontsize=24)



    




        
    




