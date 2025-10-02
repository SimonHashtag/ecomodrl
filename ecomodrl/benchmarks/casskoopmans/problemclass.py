import numpy as np

#-------------------------------------------------------------------
# Cass-Koopmans planning problem
#-------------------------------------------------------------------
class PlanningProblem():
    def __init__(
        self, 
        σ:float=2,      # Coefficient of relative risk aversion
        β:float=0.95,   # Discount factor
        δ:float=0.02,   # Depreciation rate on capital
        α:float=0.33,   # Return to capital per capita
        A:int=1):       # Technology

        self.σ, self.β = σ, β
        self.δ, self.α, self.A = δ, α, A

    def u(self, c):
        '''
        Utility function
        '''
        σ = self.σ

        return c ** (1 - σ) / (1 - σ) if σ!= 1 else np.log(c)

    def u_prime(self, c):
        'Derivative of utility'
        σ = self.σ

        return c ** (-σ)

    def u_prime_inv(self, c):
        'Inverse of derivative of utility'
        σ = self.σ

        return c ** (-1 / σ)

    def f(self, k):
        'Production function'
        α, A = self.α, self.A

        return A * k ** α

    def f_prime(self, k):
        'Derivative of production function'
        α, A = self.α, self.A

        return α * A * k ** (α - 1)

    def f_prime_inv(self, k):
        'Inverse of derivative of production function'
        α, A = self.α, self.A

        return (k / (A * α)) ** (1 / (α - 1))
    
    def cash_on_hand(self, k):
        ''''
        Given the current capital Kt, computes Kt+1 by state transition law
        assuming no consumption (for upper bound of consumption)
        '''
        δ = self.δ
        f = self.f

        k_next = f(k) + (1 - δ) * k

        return k_next
        
    def next_k(self, k, c):
        ''''
        Given the current capital Kt and an arbitrary feasible
        consumption choice Ct, computes Kt+1 by state transition law.
        '''
        k_next = self.f(k) + (1 - self.δ) * k - c
        
        return k_next

    def next_k_c(self, k, c):
        ''''
        Given the current capital Kt and an arbitrary feasible
        consumption choice Ct, computes Kt+1 by state transition law
        and optimal Ct+1 by Euler equation.
        '''
        β, δ = self.β, self.δ
        u_prime, u_prime_inv = self.u_prime, self.u_prime_inv
        f_prime = self.f_prime

        k_next = self.next_k(k, c)
        c_next = u_prime_inv(u_prime(c) / (β * (f_prime(k_next) + (1 - δ))))

        return k_next, c_next
    
    def bellman(self, c, k, value, cons_share=True):
        'Bellman equation for c = consumption share'
        if cons_share:
            c = c * self.cash_on_hand(k)
        new_k = self.next_k(k, c)
        if new_k<0:
            return np.inf
        else:
            return -(self.u(c) + self.β * value(new_k))
    
    def sq_euler_error(self, c, k, policy):
        '''
        Given current consumption and capital, computes the Euler equation error
        '''
        β, δ = self.β, self.δ
        u_prime = self.u_prime
        f_prime = self.f_prime

        k_next = self.next_k(k, c)
        if k_next < 0:
            return np.inf
        c_next = policy(k_next)

        return (β * u_prime(c_next/c) * (f_prime(k_next) + (1 - δ)) - 1)**2
    
    def euler_egm(self, c_next, k_next):
        β, δ = self.β, self.δ
        u_prime = self.u_prime
        f_prime = self.f_prime
        u_prime_inv = self.u_prime_inv
        
        return u_prime_inv(β*u_prime(c_next)*(f_prime(k_next)+(1-δ)))
    
    def calculate_euler_errors(self, c_path, k_path):
        'Calculate Euler equation errors given paths of consumption and capital'
        euler_errors = self.β * (c_path[:-1]/c_path[1:])**(self.σ) * (self.f_prime(k_path[1:-1]) + (1 - self.δ)) - 1
        
        return euler_errors