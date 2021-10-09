from math import log, sqrt, exp, pi
import scipy.stats as st
import numpy as np

class BlackScholes:

    def __init__(self, option_type, price, strike, interest_rate, expiry, volatility, dividend_yield=0, dist=st.norm):
        self.s = price  # Underlying asset price
        self.k = strike  # Option strike K
        self.r = interest_rate  # Continuous risk fee rate
        self.q = dividend_yield  # Dividend continuous rate
        self.T = expiry  # time to expiry (year)
        self.sigma = volatility  # Underlying volatility
        self.type = option_type # option type "p" put option "c" call option
        self.dist = dist # distribution to use, defaults to normal

    def n(self, d):
        # cumulative probability distribution function of standard normal distribution
        return self.dist.cdf(d)

    def dn(self, d):
        # the first order derivative of n(d)
        return self.dist.pdf(d)

    def d1(self):
        d1 = (log(self.s / self.k) + (self.r - self.q + self.sigma ** 2 * 0.5) * self.T) / (self.sigma * sqrt(self.T))
        return d1

    def d2(self):
        d2 = (log(self.s / self.k) + (self.r - self.q - self.sigma ** 2 * 0.5) * self.T) / (self.sigma * sqrt(self.T))
        return d2

    def get_price(self):
        d1 = self.d1()
        d2 = d1 - self.sigma * sqrt(self.T)
        if self.type == 'call':
            price = exp(-self.r*self.T) * (self.s * exp((self.r - self.q)*self.T) * self.n(d1) - self.k * self.n(d2))
            return price
        elif self.type == 'put':
            price = exp(-self.r*self.T) * (self.k * self.n(-d2) - (self.s * exp((self.r - self.q)*self.T) * self.n(-d1)))
            return price
        else:
            print("option type can only be call or put")

    def delta(self):
        if (self.type == 'call'):
            #delta = np.exp(-self.T) * self.n(self.d1())
            delta = self.n(self.d1())
        else:
            #delta = -np.exp(-self.T) * self.n(-self.d1())
            delta = -self.n(-self.d1())
        return delta

    def prob_itm(self):
        if (self.type == 'call'):
            itm = self.n(self.d2())
        else:
            itm = -self.n(-self.d2())
        return itm

    def theta(self):
        # https://github.com/vollib/vollib/blob/master/vollib/black_scholes/greeks/analytical.py#L96
        two_sqrt_t = 2 * np.sqrt(self.T)
        first_term = (-self.s * self.dn(self.d1()) * self.sigma) / two_sqrt_t
        if (self.type == 'call'):
            second_term = self.r * self.k * np.exp(-self.r*self.T) * self.n(self.d2())
            theta = (first_term - second_term)/365.0
        else:
            second_term = self.r * self.k * np.exp(-self.r*self.T) * self.n(-self.d2())
            theta = (first_term + second_term)/365.0
        return theta

    def vega(self):
        # S: underlying stock price # K: Option strike price # r: risk free rate # D: dividend value # vol: Volatility # T: time to expiry (assumed that we're measuring from t=0 to T)
        #  S * norm.pdf(d1) * np.sqrt(T-t)
        return self.s * self.dn(self.d1()) * sqrt(self.T)
