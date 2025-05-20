from .AbstractClass import Desision
from numpy.typing import NDArray
import numpy as np
import pandas as pd

class SimplexMethodWithoutDiscont(Desision):
    def __init__(self, P_1, P_2, R_1, R_2):
        super().__init__(P_1, P_2, R_1, R_2)
        
        self.P = [P_1, P_2]
        self.R = [R_1, R_2]
        self.strategies = []
        self.repr = None
        self.table = None
        
    def __call__(self) -> Desision:
        try:
            self.repr = ""
            n = self.P[0].shape[0]
            z = (self.__get_v() * np.ones(shape=(n, 2))).flatten()
            b = np.zeros(shape=(n,), dtype=np.float32)
            b[-1] = 1.0

            w = np.hstack((np.eye(3, dtype=np.float32), np.eye(3, dtype=np.float32)))
            w = w - np.hstack((self.P[0].T, self.P[1].T)) * np.ones(shape=(3, 6))
            W = np.zeros(shape=(3, 6))

            for k in range(3):
                W[:, 2*k] = w[:, k]
                W[:, 2*k + 1] = w[:, k + 3]

            W[-1] = np.ones(shape=(2*n, ), dtype=np.float32)
            # z[2], z[4] = z[4], z[2]
            # W[:, 2], W[:, 4] = W[:, 4].copy(), W[:, 2].copy() 
            solve, E = self.__solve_simplex(z, W, b)

            strategy = [0] * n
            for k in range(n):
                if solve[k * 2] > 0.0: strategy[k] = 1
                else: strategy[k] = 2

            self.repr += f"optim strategy: {strategy}, E={np.round(E, 2)}\n"
        except Exception as e:
            print(e)
            return self.repr
        
        return self

    def __repr__(self):
        return self.repr

    
    def __get_v(self) -> NDArray:
        n = self.P[0].shape[0]
        v = np.zeros(shape=(2, n), dtype=np.float32)
        for k in range(2):
            v[k] = np.sum(self.P[k] * self.R[k], axis=1)  
        self.repr += f"v:\n {np.round(v, 2)}\n"
                
        return v.T
    
    def __get_matrix_with_basic_vectors(self, z: NDArray, A: NDArray, b: NDArray) -> tuple:
        self.repr += "change basis:"
        M = np.hstack([A, b.reshape(-1, 1)])
        n = self.P[0].shape[0]
        self.__get_table(M, z)
        for k in range(n):
            idx = np.argmax(np.abs(M[k:, k])) + k
            idx = k
            mvalue = M[idx, k]
            M[k], M[idx] = M[idx].copy(), M[k].copy()
            M[k] = M[k] / mvalue
            for i in range(n):
                if i == k: continue
                M[i] = M[i] - M[k] * M[i, k]
                
            self.__get_table(M, z)
        
        basic_vector_idx = [k for k in range(n)]
          
        while any(M[:, -1] < 0.0):
            old_basic_idx = np.argmin(M[:, -1])
            new_basic_idx = np.argmin(M[old_basic_idx][:-1])
            M = self.__change_basis(M, old_basic_idx, new_basic_idx)
            basic_vector_idx[old_basic_idx] = new_basic_idx

        return M, basic_vector_idx
    
    def __solve_simplex(self, z: NDArray, A: NDArray, b: NDArray) -> tuple:
        N = 0
        M, basic_vector_idx = self.__get_matrix_with_basic_vectors(z, A, b)
        self.repr += f"With basic vectors:"
        self.__get_table(M, z, basic_vector_idx)
        delta = z[basic_vector_idx] @ M - np.append(z, [0.0])
        delta[basic_vector_idx] = 0.0
        self.repr += f"With delta:"
        self.__get_table(M, z, basic_vector_idx, delta=delta)
        
        while any(delta[:-1] < 0):
            b = M[:,-1]
            N += 1
            self.repr += f"Iter={N}\n"
            new_basic_idx = int(np.argmin(delta[:-1]))
            self.repr += f"min delta: idx = {new_basic_idx + 1}, delta={np.round(delta[new_basic_idx], 2)}\n"
            Q = b / M[:, new_basic_idx]
            Q[Q < 0] = np.inf
            Q[M[:, new_basic_idx] < 0] = np.inf
            old_basic_idx = int(np.argmin(Q))
            self.repr += f"min Q: idx={old_basic_idx + 1}, Q={np.round(Q[old_basic_idx], 2)}\n"
            basic_vector_idx[old_basic_idx] = new_basic_idx
            
            self.repr += f"Current table"
            self.__get_table(M, z, basic_vector_idx, delta=delta, Q=Q)
            
            M = self.__change_basis(M, old_basic_idx, new_basic_idx)
            delta = z[basic_vector_idx] @ M - np.append(z, [0.0])
            self.repr += f"Current table with update delta"
            self.__get_table(M, z, basic_vector_idx, delta, Q)
        
        b = M[:, -1]   
        solve = np.zeros(shape=(A.shape[1],))
        solve[basic_vector_idx] = b  
        F = solve @ z
        
        self.repr += f"solve={np.round(solve, 2)}, F={np.round(F, 2)}\n"
         
        return solve, F
    
    def __change_basis(self, M: NDArray, old_idx: int, new_idx: int) -> NDArray:
        M[old_idx] /= M[old_idx, new_idx]
        
        for k in range(M.shape[0]):
            if k == old_idx: continue
            M[k] -= M[old_idx]*M[k, new_idx]
            
        return M
        
    def __get_table(self, M: NDArray,
                    z: NDArray,
                    basic_vector_idx: list | None = None,
                    delta: NDArray | None = None,
                    Q: NDArray | None = None) -> None:
        n = self.P[0].shape[0]
        if delta is None:
            index = ['z']
            if basic_vector_idx is None: index += [k +1 for k in range(n)]
            else: index += [f"x{i + 1}" for i in basic_vector_idx]
            table = pd.DataFrame(
                np.round(np.vstack([np.append(z, [np.nan]), M]), 4),
                columns=[f"x{i + 1}" for i in range(2*n)] + ['b'],
                index=index
            )
        else:
            if Q is None:
                table = pd.DataFrame(
                    np.round(np.vstack([np.append(z, [np.nan]), M, delta]), 4),
                    columns=[f"x{i + 1}" for i in range(2*n)] + ['b'],
                    index=['z'] + [f"x{i + 1}" for i in basic_vector_idx] + ['delta']
                )
            else:
                table = pd.DataFrame(
                    np.round(np.hstack([
                        np.vstack([
                            np.append(z, [np.nan]),
                            M, delta]),
                        np.concatenate(([np.nan], Q, [np.nan])).reshape(-1, 1)]), 4),
                    columns=[f"x{i + 1}" for i in range(2*n)] + ['b'] + ['Q'],
                    index=['z'] + [f"x{i + 1}" for i in basic_vector_idx] + ['delta']
                )
        self.repr += f"\n{table}\n"
        
        

class SimplexMethodWithDiscont(Desision):
    def __init__(self, P_1, P_2, R_1, R_2):
        super().__init__(P_1, P_2, R_1, R_2)
        
        self.P = [P_1, P_2]
        self.R = [R_1, R_2]
        self.strategies = []
        self.repr = None
        self.table = None
        
    def __call__(self, b: NDArray, alpha: float) -> Desision:
        self.repr = ""
        n = self.P[0].shape[0]
        z = (self.__get_v() * np.ones(shape=(n, 2))).flatten()
        w = np.hstack((np.eye(3, dtype=np.float32), np.eye(3, dtype=np.float32)))
        w = w - alpha * np.hstack((self.P[0].T, self.P[1].T)) * np.ones(shape=(3, 6))
        W = np.zeros(shape=(3, 6))
        for k in range(3):
            W[:, 2*k] = w[:, k]
            W[:, 2*k + 1] = w[:, k + 3]
        solve, E = self.__solve_simplex(z, W, b)
        strategy = [0] * n
        for k in range(n):
            if solve[k * 2] > 0.0: strategy[k] = 1
            else: strategy[k] = 2
        self.repr += f"optim strategy: {strategy}, E={np.round(E, 2)}\n"
        
        return self

    def __repr__(self):
        return self.repr

    
    def __get_v(self) -> NDArray:
        n = self.P[0].shape[0]
        v = np.zeros(shape=(2, n), dtype=np.float32)
        for k in range(2):
            v[k] = np.sum(self.P[k] * self.R[k], axis=1)  
        self.repr += f"v:\n {np.round(v, 2)}\n"
                
        return v.T
    
    def __get_matrix_with_basic_vectors(self, z: NDArray, A: NDArray, b: NDArray) -> tuple:
        self.repr += "change basis:"
        M = np.hstack([A, b.reshape(-1, 1)])
        n = self.P[0].shape[0]
        self.__get_table(M, z)
        for k in range(n):
            idx = np.argmax(np.abs(M[k:, k])) + k
            idx = k
            mvalue = M[idx, k]
            M[k], M[idx] = M[idx].copy(), M[k].copy()
            M[k] = M[k] / mvalue
            for i in range(n):
                if i == k: continue
                M[i] = M[i] - M[k] * M[i, k]
                
            self.__get_table(M, z)
        
        basic_vector_idx = [k for k in range(n)]
          
        while any(M[:, -1] < 0.0):
            old_basic_idx = int(np.argmin(M[:, -1]))
            new_basic_idx = int(np.argmin(M[old_basic_idx][:-1]))
            M = self.__change_basis(M, old_basic_idx, new_basic_idx)
            basic_vector_idx[old_basic_idx] = new_basic_idx

        return M, basic_vector_idx
    
    def __solve_simplex(self, z: NDArray, A: NDArray, b: NDArray) -> tuple:
        N = 0
        M, basic_vector_idx = self.__get_matrix_with_basic_vectors(z, A, b)
        self.repr += f"With basic vectors:"
        self.__get_table(M, z, basic_vector_idx)
        delta = z[basic_vector_idx] @ M - np.append(z, [0.0])
        delta[basic_vector_idx] = 0.0
        self.repr += f"With delta:"
        self.__get_table(M, z, basic_vector_idx, delta=delta)
        
        while any(delta < 0):
            b = M[:,-1]
            N += 1
            self.repr += f"Iter={N}\n"
            new_basic_idx = int(np.argmin(delta))
            self.repr += f"min delta: idx = {new_basic_idx + 1}, delta={np.round(delta[new_basic_idx], 2)}\n"
            Q = b / M[:, new_basic_idx]
            Q[Q < 0] = np.inf
            Q[M[:, new_basic_idx] < 0] = np.inf
            old_basic_idx = int(np.argmin(Q))
            self.repr += f"min Q: idx={old_basic_idx + 1}, Q={np.round(Q[old_basic_idx], 2)}\n"
            basic_vector_idx[old_basic_idx] = new_basic_idx
            
            self.repr += f"Current table"
            self.__get_table(M, z, basic_vector_idx, delta=delta, Q=Q)
            
            M = self.__change_basis(M, old_basic_idx, new_basic_idx)
            delta = z[basic_vector_idx] @ M - np.append(z, [0.0])
            self.repr += f"Current table with update delta"
            self.__get_table(M, z, basic_vector_idx, delta, Q)
        
        b = M[:, -1]   
        solve = np.zeros(shape=(A.shape[1],))
        solve[basic_vector_idx] = b  
        F = solve @ z
        
        self.repr += f"solve={np.round(solve, 2)}, F={np.round(F, 2)}\n"
         
        return solve, F
    
    def __change_basis(self, M: NDArray, old_idx: int, new_idx: int) -> NDArray:
        M[old_idx] /= M[old_idx, new_idx]
        
        for k in range(M.shape[0]):
            if k == old_idx: continue
            M[k] -= M[old_idx]*M[k, new_idx]
            
        return M
        
    def __get_table(self, M: NDArray,
                    z: NDArray,
                    basic_vector_idx: list | None = None,
                    delta: NDArray | None = None,
                    Q: NDArray | None = None) -> None:
        n = self.P[0].shape[0]
        if delta is None:
            index = ['z']
            if basic_vector_idx is None: index += [k +1 for k in range(n)]
            else: index += [f"x{i + 1}" for i in basic_vector_idx]
            table = pd.DataFrame(
                np.round(np.vstack([np.append(z, [np.nan]), M]), 4),
                columns=[f"x{i + 1}" for i in range(2*n)] + ['b'],
                index=index
            )
        else:
            if Q is None:
                table = pd.DataFrame(
                    np.round(np.vstack([np.append(z, [np.nan]), M, delta]), 4),
                    columns=[f"x{i + 1}" for i in range(2*n)] + ['b'],
                    index=['z'] + [f"x{i + 1}" for i in basic_vector_idx] + ['delta']
                )
            else:
                table = pd.DataFrame(
                    np.round(np.hstack([
                        np.vstack([
                            np.append(z, [np.nan]),
                            M, delta]),
                        np.concatenate(([np.nan], Q, [np.nan])).reshape(-1, 1)]), 4),
                    columns=[f"x{i + 1}" for i in range(2*n)] + ['b'] + ['Q'],
                    index=['z'] + [f"x{i + 1}" for i in basic_vector_idx] + ['delta']
                )
        self.repr += f"\n{table}\n"
        
        
        
        