import numpy as np
from scipy.interpolate import CubicSpline
from typing import Union

def inverse_cubic_interpolation(fn, x:np.ndarray, extrapolate:bool=True):
    """Inverse interpolation via cubic splines"""
    cs = CubicSpline(fn(x), x, bc_type="natural", extrapolate=extrapolate)
    return cs

def wev(x:Union[np.ndarray, float], V_bm, V) -> Union[np.ndarray, float]:
    """
    Wealth equivalent variation (WEV)
    
    x: the capital level(s) at which to evaluate WEV
    V_bm: benchmark value function (e.g. from VFI)  
    V: value function from RL agent
    
    Interpretation 1: The additional capital that would make the agent indifferent between the two value functions
                    If RL is worse than benchmark, WEV > 0 (need more capital to be as well off as with benchmark)
    Interpretation 2: Distance between the two value functions in terms of capital (cash-on-hand)
    """
    V_inv = inverse_cubic_interpolation(V, np.sort(x))
    return V_inv(V_bm(x)) - x

def cev_crra(states:Union[np.ndarray, float], V_bm, V, σ:float) -> Union[np.ndarray, float]:
    """Consumption-equivalent variation (CEV)"""
    return (V_bm(states)/V(states))**(1/(1-σ)) - 1