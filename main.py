import numpy as np
import pandas as pd
from Desisions import (DPEnumerationN,
                       DPEnumerationInf,
                       DPIterationWithoutDiscont,
                       DPIterationWithDiscont,
                       SimplexMethodWithoutDiscont,
                       SimplexMethodWithDiscont)

def get_real():
    P_1 = np.array(
        [[0.5, 0.3, 0.2],
        [0.6, 0.3, 0.1],
        [0.1, 0.3, 0.6]],
        dtype=np.float32)

    P_2 = np.array(
        [[0.3, 0.3, 0.4],
         [0.1, 0.4, 0.5],
         [0.1, 0.1, 0.8]],
        dtype=np.float32)

    R_1 = np.array(
        [[3, 2, -1],
         [1, 1, -3],
         [2, 0, -5]],
        dtype=np.int32)

    R_2 = np.array(
        [[4, 1, 0],
         [2, 2, -1],
         [3, 1, -3]],
        dtype=np.int32)
    
    return P_1, P_2, R_1, R_2

def get_test():
    P_1 = np.array(
        [[0.2, 0.5, 0.3],
        [0.0, 0.5, 0.5],
        [0.0, 0.0, 1.0]],
        dtype=np.float32)

    P_2 = np.array(
        [[0.3, 0.6, 0.1],
         [0.1, 0.6, 0.3],
         [0.05, 0.4, 0.55]],
        dtype=np.float32)

    R_1 = np.array(
        [[7, 6, 3],
         [0, 5, 1],
         [0, 0, -1]],
        dtype=np.int32)

    R_2 = np.array(
        [[6, 5, -1],
         [7, 4, 0],
         [6, 3, -2]],
        dtype=np.int32)
    return P_1, P_2, R_1, R_2

if __name__ == "__main__":
    P_1, P_2, R_1, R_2 = get_real()
    assignments = {
        "Enumeration N":DPEnumerationN,
        "Enumeration Inf": DPEnumerationInf,
        "Iteration without Discont": DPIterationWithoutDiscont,
        "Iteration with Discont": DPIterationWithDiscont,
        "Simplex method without Discont": SimplexMethodWithoutDiscont,
        "Simplex method with Discont": SimplexMethodWithDiscont
    }
    input_values = {
        "Enumeration N": [3],
        "Enumeration Inf": None,
        "Iteration without Discont": None,
        "Iteration with Discont": [0.7],
        "Simplex method without Discont": None,
        "Simplex method with Discont": [np.array([1.0, 1.0, 1.0]), 0.7]
    }
    
    for idx, (name, decision) in enumerate(assignments.items()):
        print(f"Assignment {idx + 1}: {name}")

        args = input_values[name]
        if args is None:
            result = decision(P_1, P_2, R_1, R_2)()
        else:
            result = decision(P_1, P_2, R_1, R_2)(*args)
        print(result)
        print("----------------------------------------------------------")
        