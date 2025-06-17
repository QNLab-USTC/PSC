import sys
import os

sys.setrecursionlimit(100000)  # 例如这里设置为十万
for i in range(200):
    print(f"#################################{i}##############################################")
    print("Task 1!")
    os.system("python quantum_routing_failure.py")
    print("Task 2!")
    os.system("python relax_problem_program.py")
    print("Task 3!")
    os.system("python heuristic_algorithm.py")
    print(f"#################################{i}##############################################")
