from abc import ABC, abstractmethod
from numpy.typing import NDArray

class Desision(ABC):
    @abstractmethod
    def __init__(self,
                 P_1: NDArray, P_2: NDArray,
                 R_1: NDArray, R_2: NDArray):
        pass 
    
    @abstractmethod
    def __call__(self, *args, **kwds):
        pass
    
    @abstractmethod
    def __repr__(self):
        pass