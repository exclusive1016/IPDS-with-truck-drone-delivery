# IPDS-with-truck-drone-delivery
Code and computational results for the paper "Integrated Production and Parallel Truck-Drone Delivery Scheduling to Minimize the Makespan: Complexity and Algorithms"

## 1. Repository Structure
- `Instances/`: Scripts used to generate the test instances and the detailed instance data used in our numerical experiments.
- `MIP/`: Exact mixed-integer programming (MIP) models implemented with Gurobi, along with the computational results for P3, P4, and the correlated instances.
- `Heuristic/`: Implementations and computational results for the proposed approximation algorithms, as well as the comparative baseline methods (Truck-Only and Max-Drone).

## 2. Dependencies
To run the code, the following environment and packages are required:
- Python 3.10+
- Gurobi Optimizer (with a valid academic license)
- `numpy`, `pandas`, `openpyxl`, `gurobipy`

## 3. How to Run
Follow these general steps to reproduce the computational results:

**Test Instances:**
The specific test instances are already provided in the `Instances/` folder. You can also run the generation script within this folder to create new datasets.

**Run Exact Models:**
Navigate to the `MIP/` folder and run the corresponding script to obtain the exact lower bounds or optimal solutions using Gurobi.

**Run Approximation Algorithms:**
Navigate to the `Heuristic/` folder and execute the algorithm scripts to obtain the makespan results for the proposed approximation approaches and baselines. 

## 4. Citation
If you find our code or paper useful, please consider citing it (citation information will be updated upon publication).
