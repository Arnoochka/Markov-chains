from .AbstractClass import Desision
from numpy.typing import NDArray
import numpy as np
import pandas as pd

class DPEnumerationN(Desision):
    def __init__(self,
                 P_1: NDArray, P_2: NDArray,
                 R_1: NDArray, R_2: NDArray):
        super().__init__(P_1, P_2, R_1, R_2) 
        
        self.P = [P_1, P_2]
        self.R = [R_1, R_2]     
        self.repr = None
        self.table = None
        
    def __call__(self, N: int, alpha: float = 1.0) -> Desision:
        self.repr  = ""
        self.table = None
        
        self.v = self.__get_v()
        f = self.__get_f(self.v)
        self.__get_table(self.v, f, N)
        
        for n in range(N - 1, 0, -1):
            f = self.__step(f, n, alpha)
        
        return self
    
    def __repr__(self) -> str:   
        return self.repr

    def __step(self, f: NDArray, N: int, alpha: float) -> NDArray:
        n = self.P[0].shape[0]
        prom = self.v.copy()
        
        for k in range(2):
            for i in range(n):
                prom[k][i] += alpha * (self.P[k][i] @ f)
                
        f = self.__get_f(prom)
        self.__get_table(prom, f, N)
        
        return f
        
    def __get_v(self) -> NDArray:
        n = self.P[0].shape[0]
        v = np.zeros(shape=(2, n), dtype=np.float32)
        for k in range(2):
            v[k] = np.sum(self.P[k] * self.R[k], axis=1)
        self.repr += f"v\n {np.round(v, 2)}\n"
                
        return v
    
    def __get_f(self, prom: NDArray) -> NDArray:
        f = np.array(
            [np.max(row) for row in prom.T]
        )
        
        return f
    
    def __get_table(self, prom: NDArray, f: NDArray, N: int) -> pd.DataFrame:
        n = self.P[0].shape[0]
        k_opt = np.array(
            [np.argmax(row) + 1 for row in prom.T]
        )
        table = np.column_stack((prom[0], prom[1], f, k_opt))
        step = pd.DataFrame(
                table,
                index=range(1, n+1),
                columns=["k=1", "k=2", "f", "k optim"]
            )
        self.repr += f"step={N}\n{step.round(2)}\n"
        if N == 1:
            self.table = step
            self.best_desision = round(float(np.max(f)), 2)
            self.repr += f"\nMaximum expected revenue: {self.best_desision}, table:\n{self.table.round(2)}"
        return step
    
class DPEnumerationInf(Desision):
    def __init__(self, P_1, P_2, R_1, R_2):
        super().__init__(P_1, P_2, R_1, R_2)
        
        self.P = [P_1, P_2]
        self.R = [R_1, R_2]
        self.strategies = []
        self.repr = None
        self.table = None
        
    def __call__(self) -> Desision:
        self.repr = ""
        self.table = []
        
        n = self.P[0].shape[0]
        
        for s in range(2**n):
            v, P, R, binary = self.__get_strategy(s)
            self.repr += (f"s: {s+1}, {'0' * (n - len(binary)) + binary}\n"
                          f" v: {np.round(v, 2)}\n P\n {np.round(P, 2)}\n R\n {np.round(R, 2)}\n")
            solve = self.__solve_linears_system(P)
            E = solve @ v
            self.repr += f" E: {np.round(E, 2)}\n"
            
            self.table.append(['0' * (n - len(binary)) + binary,
                               np.round(v, 2),
                               np.round(solve, 2),
                               np.round(E, 2)])
            
        self.table = pd.DataFrame(
            self.table, 
            columns=["binary", "v", "solve", "E"],
            index=range(1, 2**n + 1)
        )
        
        self.repr += f"\nbest strategy: {np.argmax(self.table['E']) + 1}, E={np.round(np.max(self.table['E']), 2)} Table\n{self.table.round(2)}"
         
        return self
    
    def __repr__(self) -> str:
        return self.repr
    
    def __get_strategy(self, s: int) -> tuple:
        n = self.P[0].shape[0]
        binary_str = bin(s)[2:] 
        binary = [0] * (n - len(binary_str)) + [int(num) for num in binary_str]
        P = np.array(
            [self.P[val][idx] for idx, val in enumerate(binary)])
        R = np.array(
            [self.R[val][idx] for idx, val in enumerate(binary)])
        
        v = np.sum(P * R, axis=1)
        return v, P, R, binary_str
            
    def __solve_linears_system(self, P: NDArray) -> NDArray:
        n = P.shape[0]
        A = np.vstack(
            [np.delete(P.T - np.eye(n), n - 1, axis=0),
            np.ones(shape=(n, ))])
        b = np.zeros(shape=(n, ))
        b[-1] = 1.0
        solve = np.linalg.solve(A, b)
        self.repr += f" System:\n A\n {np.round(A, 2)}\n b\n {np.round(b, 2)}\n solve: {np.round(solve, 2)}\n"
        return solve