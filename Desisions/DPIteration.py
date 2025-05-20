from .AbstractClass import Desision
from numpy.typing import NDArray
import numpy as np
import pandas as pd

class DPIterationWithoutDiscont(Desision):
    def __init__(self, P_1, P_2, R_1, R_2):
        super().__init__(P_1, P_2, R_1, R_2)
        
        self.P = [P_1, P_2]
        self.R = [R_1, R_2]
        self.strategies = []
        self.repr = None
        self.table = None
        
    def __call__(self, strategy: list[int] | None = None) -> Desision:
        self.repr = ""
        self.table = []
        n = self.P[0].shape[0]
        N = 0
        if strategy is None: 
            strategy = [0] * n
            
        v_original = self.__get_v_original()
            
        while True:
            N += 1
            v, P, R, binary_str = self.__get_strategy(strategy, N)  
            E, solve = self.__solve_linears_system(P, v)  
            self.table.append([N, binary_str, E])
            new_strategy = self.__step(solve, v_original, N)
            
            if strategy != new_strategy:
                strategy = new_strategy.copy()
            else: break
          
        self.table = pd.DataFrame(self.table,
                                  columns=["Iter", "Strategy", "E"],
                                  index=range(1, N+1))  
        
        self.repr += f"\nBest strategy: {''.join(map(str,strategy))}, E={np.round(E, 2)}, Table:\n {self.table.round(2)}"
          
        return self

    def __repr__(self):
        return self.repr
    
    def __get_strategy(self, binary: list[int], N: int) -> tuple:
        n = self.P[0].shape[0]
        P = np.array(
            [self.P[val][idx] for idx, val in enumerate(binary)])
        R = np.array(
            [self.R[val][idx] for idx, val in enumerate(binary)])
        
        v = np.sum(P * R, axis=1)
        binary_str = ''.join(map(str, binary))
        
        self.repr += (f"Iter={N}, strategy: {binary_str}\n"
                      f" v: {np.round(v, 2)}\n P\n {np.round(P, 2)}\n R\n {np.round(R, 2)}\n")
        
        return v, P, R, binary_str
    
    def __solve_linears_system(self, P: NDArray, v: NDArray) -> NDArray:
        n = P.shape[0]
        A = (np.eye(n) - P)
        A[:, -1] = np.ones(shape=(n,))
        b = v
        solve = np.linalg.solve(A, b)
        E = solve[-1]
        solve[-1] = 0.0
        self.repr += f" System:\n A\n {np.round(A, 2)}\n b\n {np.round(b, 2)}\n solve: {np.round(solve, 2)}, E={np.round(E, 2)}\n"
        return E, solve   
    
    def __step(self, f: NDArray, v: NDArray, N: int) -> list:
        n = self.P[0].shape[0]
        prom = v.copy()
        
        for k in range(2):
            for i in range(n):
                prom[k][i] += (self.P[k][i] @ f)
                
        f = self.__get_f(prom)
        new_strategy = self.__get_new_strategy(prom, f)
        
        return new_strategy
    
    def __get_v_original(self) -> NDArray:
        n = self.P[0].shape[0]
        v = np.zeros(shape=(2, n), dtype=np.float32)
        for k in range(2):
            v[k] = np.sum(self.P[k] * self.R[k], axis=1)  
        self.repr += f"v original:\n {np.round(v, 2)}\n"
                
        return v
    
    def __get_new_strategy(self, prom: NDArray, f: NDArray) -> list:
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
        self.repr += f" {step.round(2)}\n"
        return list(int(s) for s in (k_opt - 1))
    
    def __get_f(self, prom: NDArray) -> NDArray:
        f = np.array(
            [np.max(row) for row in prom.T]
        )
        
        return f
    
class DPIterationWithDiscont(Desision):
    def __init__(self, P_1, P_2, R_1, R_2):
        super().__init__(P_1, P_2, R_1, R_2)
        
        self.P = [P_1, P_2]
        self.R = [R_1, R_2]
        self.strategies = []
        self.repr = None
        self.table = None
        
    def __call__(self, alpha: float, strategy: list[int] | None = None) -> Desision:
        self.repr = ""
        self.table = []
        n = self.P[0].shape[0]
        N = 0
        if strategy is None: 
            strategy = [0] * n
            
        v_original = self.__get_v_original()
            
        while True:
            N += 1
            v, P, R, binary_str = self.__get_strategy(strategy, N)  
            solve = self.__solve_linears_system(P, v, alpha)  
            self.table.append([N, binary_str])
            new_strategy, f = self.__step(solve, v_original, N, alpha)
            
            if strategy != new_strategy:
                strategy = new_strategy.copy()
            else: break
          
        self.table = pd.DataFrame(self.table,
                                  columns=["Iter", "Strategy"],
                                  index=range(1, N+1))  
        
        self.repr += f"\nBest strategy: {''.join(map(str,strategy))}, Maximum expected revenue: {round(float(np.sum(f)), 2)}, Table:\n {self.table.round(2)}"
          
        return self

    def __repr__(self):
        return self.repr
    
    def __get_strategy(self, binary: list[int], N: int) -> tuple:
        n = self.P[0].shape[0]
        P = np.array(
            [self.P[val][idx] for idx, val in enumerate(binary)])
        R = np.array(
            [self.R[val][idx] for idx, val in enumerate(binary)])
        
        v = np.sum(P * R, axis=1)
        binary_str = ''.join(map(str, binary))
        
        self.repr += (f"Iter={N}, strategy: {binary_str}\n"
                      f" v: {np.round(v, 2)}\n P\n {np.round(P, 2)}\n R\n {np.round(R, 2)}\n")
        
        return v, P, R, binary_str
    
    def __solve_linears_system(self, P: NDArray, v: NDArray, alpha: float) -> NDArray:
        n = P.shape[0]
        A = (np.eye(n) - alpha * P)
        b = v
        solve = np.linalg.solve(A, b)
        self.repr += f" System:\n A\n {np.round(A, 2)}\n b\n {np.round(b, 2)}\n solve: {np.round(solve, 2)}\n"
        return solve   
    
    def __step(self, f: NDArray, v: NDArray, N: int, alpha: float) -> list:
        n = self.P[0].shape[0]
        prom = v.copy()
        
        for k in range(2):
            for i in range(n):
                prom[k][i] += alpha * (self.P[k][i] @ f)
                
        f = self.__get_f(prom)
        new_strategy = self.__get_new_strategy(prom, f)
        
        return new_strategy, f
    
    def __get_v_original(self) -> NDArray:
        n = self.P[0].shape[0]
        v = np.zeros(shape=(2, n), dtype=np.float32)
        for k in range(2):
            v[k] = np.sum(self.P[k] * self.R[k], axis=1)  
        self.repr += f"v original:\n {np.round(v, 2)}\n"
                
        return v
    
    def __get_new_strategy(self, prom: NDArray, f: NDArray) -> list:
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
        self.repr += f" {step.round(2)}\n"
        return list(int(s) for s in (k_opt - 1))
    
    def __get_f(self, prom: NDArray) -> NDArray:
        f = np.array(
            [np.max(row) for row in prom.T]
        )
        
        return f
        
    