import gurobipy as gp
from gurobipy import GRB
import json
import csv
import time
import math
from multiprocessing import Pool, cpu_count
import os


# ==========================================
# 1. 启发式算法所需的基础函数
# ==========================================
def FFD_packing(jobs, W_t=1.0):
    sorted_jobs = sorted(jobs, key=lambda x: x['w'], reverse=True)
    batches = []
    for job in sorted_jobs:
        placed = False
        for b in batches:
            if round(sum(j['w'] for j in b) + job['w'], 4) <= W_t:
                b.append(job)
                placed = True
                break
        if not placed:
            batches.append([job])
    return batches


def NFD_packing(jobs, W_t=1.0):
    sorted_jobs = sorted(jobs, key=lambda x: x['w'], reverse=True)
    batches = []
    if not sorted_jobs: return batches
    current_batch = [sorted_jobs[0]]
    current_w = sorted_jobs[0]['w']
    for job in sorted_jobs[1:]:
        if round(current_w + job['w'], 4) <= W_t:
            current_batch.append(job)
            current_w += job['w']
        else:
            batches.append(current_batch)
            current_batch = [job]
            current_w = job['w']
    if current_batch:
        batches.append(current_batch)
    return batches


def Procedure_OS(truck_batches, drone_jobs, T_t, T_d):
    tb_info = [{'batch': b, 'P': sum(j['p'] for j in b)} for b in truck_batches]
    tb_info.sort(key=lambda x: x['P'])
    dj_info = [{'job': j, 'P': j['p']} for j in drone_jobs]
    dj_info.sort(key=lambda x: x['P'])

    b_count = len(tb_info)
    for i, tb in enumerate(tb_info): tb['q'] = (b_count - i) * T_t
    l_count = len(dj_info)
    for j, dj in enumerate(dj_info): dj['q'] = (l_count - j) * T_d

    all_tasks = tb_info + dj_info
    all_tasks.sort(key=lambda x: x['q'], reverse=True)

    current_C, makespan = 0, 0
    for task in all_tasks:
        current_C += task['P']
        makespan = max(makespan, current_C + task['q'])
    return makespan


def Baseline_Truck_Only(jobs, T_t, T_d):
    return Procedure_OS(FFD_packing(jobs), [], T_t, T_d)


def Baseline_Max_Drone(jobs, W_d, T_t, T_d):
    drone_jobs = [j for j in jobs if j['w'] <= W_d]
    truck_batches = FFD_packing([j for j in jobs if j['w'] > W_d])
    return Procedure_OS(truck_batches, drone_jobs, T_t, T_d)


def Algorithm_Proposed_OS(jobs, W_d, T_t, T_d, pack_rule='NFD'):
    J_t = [j for j in jobs if j['w'] > W_d]
    J_d_cand = [j for j in jobs if j['w'] <= W_d]
    J_d_cand.sort(key=lambda x: x['w'], reverse=True)

    best_makespan = float('inf')
    l_range = range(1, len(J_d_cand) + 1) if len(J_d_cand) > 0 else [0]

    for l in l_range:
        drone_jobs = J_d_cand[:l]
        truck_jobs = J_t + J_d_cand[l:]
        truck_batches = FFD_packing(truck_jobs) if pack_rule == 'FFD' else NFD_packing(truck_jobs)
        mksp = Procedure_OS(truck_batches, drone_jobs, T_t, T_d)
        if mksp < best_makespan:
            best_makespan = mksp
    return best_makespan


# ==========================================
# 2. 注入专用函数: 生成 Warm Start 决策变量
# ==========================================
def generate_ffd_os_warm_start(jobs_data, W_t, W_d, T_t, T_d):
    jobs = list(jobs_data.keys())
    J_t = [j for j in jobs if jobs_data[j]['w'] > W_d]
    J_d_cand = [j for j in jobs if jobs_data[j]['w'] <= W_d]
    J_d_cand.sort(key=lambda j: jobs_data[j]['w'], reverse=True)

    best_makespan = float('inf')
    best_assignment = {'x': {}, 'y': {}, 'z': {}}
    l_range = range(1, len(J_d_cand) + 1) if len(J_d_cand) > 0 else [0]

    for l in l_range:
        drone_jobs = J_d_cand[:l]
        truck_jobs = J_t + J_d_cand[l:]
        truck_jobs.sort(key=lambda j: jobs_data[j]['w'], reverse=True)

        batches = []
        for j in truck_jobs:
            placed = False
            for b in batches:
                if round(sum(jobs_data[k]['w'] for k in b) + jobs_data[j]['w'], 4) <= W_t:
                    b.append(j)
                    placed = True
                    break
            if not placed:
                batches.append([j])

        tb_P = [sum(jobs_data[k]['p'] for k in b) for b in batches]
        tb_P.sort()
        dj_P = [jobs_data[k]['p'] for k in drone_jobs]
        dj_P.sort()

        tasks = []
        b_count = len(tb_P)
        for i, p in enumerate(tb_P): tasks.append({'P': p, 'q': (b_count - i) * T_t})
        l_count = len(dj_P)
        for j, p in enumerate(dj_P): tasks.append({'P': p, 'q': (l_count - j) * T_d})
        tasks.sort(key=lambda x: x['q'], reverse=True)

        current_C = 0
        makespan = 0
        for task in tasks:
            current_C += task['P']
            makespan = max(makespan, current_C + task['q'])

        if makespan < best_makespan:
            best_makespan = makespan
            x_start = {(j, idx): 0 for j in jobs for idx in range(len(jobs))}
            y_start = {idx: 0 for idx in range(len(jobs))}
            z_start = {j: 0 for j in jobs}
            for j in drone_jobs: z_start[j] = 1
            for idx, b in enumerate(batches):
                y_start[idx] = 1
                for j in b: x_start[j, idx] = 1
            best_assignment = {'x': x_start, 'y': y_start, 'z': z_start}

    return best_makespan, best_assignment['x'], best_assignment['y'], best_assignment['z']


# ==========================================
# 3. 求解核心函数 (单次任务)
# ==========================================
def solve_instance_worker(args):
    instance_data, time_limit, threads_limit = args

    inst_id = instance_data.get('Instance_ID', 0)
    n = instance_data['n']
    dist_type = instance_data.get('Distribution', instance_data.get('Weight_Distribution', 'Unknown'))
    W_t = 1.0
    W_d, T_t, T_d, v_ratio = instance_data['Wd'], instance_data['T_t'], instance_data['T_d'], instance_data['v']

    p_list, w_list = instance_data['p_j'], instance_data['w_j']
    p = {j: p_list[j] for j in range(n)}
    w = {j: w_list[j] for j in range(n)}

    jobs_data = {j: {'p': p[j], 'w': w[j]} for j in range(n)}
    jobs = list(range(n))
    batches = list(range(n))
    heavy_jobs = [j for j in jobs if w[j] > W_d]
    M = sum(p.values()) + n * T_t
    LB_theo = sum(p.values()) + min(T_t, T_d)

    # ==========================================
    # 先行计算所有启发式的 Makespan
    # ==========================================
    jobs_for_heur = [{'id': j, 'p': p[j], 'w': w[j]} for j in range(n)]

    mksp_truck = Baseline_Truck_Only(jobs_for_heur, T_t, T_d)
    mksp_maxdrone = Baseline_Max_Drone(jobs_for_heur, W_d, T_t, T_d)
    mksp_nfd = Algorithm_Proposed_OS(jobs_for_heur, W_d, T_t, T_d, pack_rule='NFD')

    # 获取 FFD+OS 的 Makespan 及其初始解
    mksp_ffd, x_start, y_start, z_start = generate_ffd_os_warm_start(jobs_data, W_t, W_d, T_t, T_d)

    # 建立模型
    unique_name = f"TD_{n}_{W_d}_{v_ratio}_{inst_id}"
    model = gp.Model(unique_name)
    model.setParam('OutputFlag', 0)
    model.setParam('TimeLimit', time_limit)
    model.setParam('Threads', threads_limit)
    model.setParam('MIPGap', 0.005)

    # --- 变量与约束 ---
    C = model.addVars(jobs, vtype=GRB.CONTINUOUS)
    St = model.addVars(batches, vtype=GRB.CONTINUOUS)
    Sd = model.addVars(jobs, vtype=GRB.CONTINUOUS)
    D_max = model.addVar(vtype=GRB.CONTINUOUS)
    sigma = model.addVars(jobs, jobs, vtype=GRB.BINARY)
    x = model.addVars(jobs, batches, vtype=GRB.BINARY)
    z = model.addVars(jobs, vtype=GRB.BINARY)
    y = model.addVars(batches, vtype=GRB.BINARY)

    for j in jobs: model.addConstr(C[j] >= p[j])
    for j in jobs:
        for k in jobs:
            if j != k:
                model.addConstr(C[j] <= C[k] - p[k] + M * (1 - sigma[j, k]))
                if j < k: model.addConstr(sigma[j, k] + sigma[k, j] == 1)
    for j in jobs: model.addConstr(z[j] + x.sum(j, '*') == 1)
    for j in heavy_jobs: model.addConstr(z[j] == 0)
    for i in batches:
        model.addConstr(gp.quicksum(w[j] * x[j, i] for j in jobs) <= W_t * y[i])
        model.addConstr(gp.quicksum(x[j, i] for j in jobs) >= y[i])
    for i in range(1, n): model.addConstr(y[i] <= y[i - 1])
    for j in jobs:
        for i in batches: model.addConstr(St[i] >= C[j] - M * (1 - x[j, i]))
    for i in range(1, n): model.addConstr(St[i] >= St[i - 1] + T_t - M * (1 - y[i]))
    for j in jobs: model.addConstr(Sd[j] >= C[j] - M * (1 - z[j]))
    for j in jobs:
        for k in jobs:
            if j != k: model.addConstr(Sd[k] >= Sd[j] + T_d - M * (3 - z[k] - z[j] - sigma[j, k]))
    for i in batches: model.addConstr(D_max >= St[i] + T_t - M * (1 - y[i]))
    for j in jobs: model.addConstr(D_max >= Sd[j] + T_d - M * (1 - z[j]))
    for i in batches: model.addConstr(St[i] <= M * y[i])
    for j in jobs: model.addConstr(Sd[j] <= M * z[j])

    heavy_weight_sum = sum(w[j] for j in heavy_jobs)
    model.addConstr(y.sum() >= math.ceil(heavy_weight_sum / W_t))
    model.addConstr(W_t * y.sum() >= sum(w.values()) - n * W_d)

    # 注入 Warm Start
    for j in jobs: z[j].Start = z_start[j]
    for i in batches: y[i].Start = y_start[i]
    for j in jobs:
        for i in batches: x[j, i].Start = x_start[j, i]

    # 求解
    model.setObjective(D_max, GRB.MINIMIZE)
    start_time = time.time()
    model.optimize()
    solve_time = time.time() - start_time

    # --- 结果打包 (大一统字典) ---
    res = {
        "Instance_ID": inst_id,
        "Distribution": dist_type,
        "n": n,
        "W_d": W_d,
        "v": v_ratio,
        "Status": model.status,

        "Mksp_Truck": mksp_truck,
        "Mksp_MaxDrone": mksp_maxdrone,
        "Mksp_NFD_OS": mksp_nfd,
        "Mksp_FFD_OS": mksp_ffd,

        "Obj_Gurobi": float('inf'),
        "LB_Gurobi": 0.0,
        "LB_Theo": round(LB_theo, 2),
        "LB_Best": round(LB_theo, 2),
        "MIP_Gap(%)": 100.0,
        "Time(s)": round(solve_time, 2),

        "Gap_Truck(%)": 0.0,
        "Gap_MaxDrone(%)": 0.0,
        "Gap_NFD_OS(%)": 0.0,
        "Gap_FFD_OS(%)": 0.0
    }

    # 1. 提取 Gurobi 目标值和 Gap（仅在找到解时有效）
    if model.SolCount > 0:
        res["Obj_Gurobi"] = round(model.ObjVal, 2)
        res["MIP_Gap(%)"] = round(model.MIPGap * 100, 2)
    else:
        res["Obj_Gurobi"] = float('inf')
        res["MIP_Gap(%)"] = 100.0

    # 2. 提取下界（无论是否找到可行解，Gurobi 都有下界 ObjBound）
    res["LB_Gurobi"] = round(model.ObjBound, 2)
    res["LB_Best"] = max(res["LB_Theo"], res["LB_Gurobi"])

    # 智能双基准逻辑
    base_val = res["Obj_Gurobi"] if n == 20 else res["LB_Best"]

    res["Gap_Truck(%)"] = round((mksp_truck - base_val) / base_val * 100, 2)
    res["Gap_MaxDrone(%)"] = round((mksp_maxdrone - base_val) / base_val * 100, 2)
    res["Gap_NFD_OS(%)"] = round((mksp_nfd - base_val) / base_val * 100, 2)
    res["Gap_FFD_OS(%)"] = round((mksp_ffd - base_val) / base_val * 100, 2)

    return res


# ==========================================
# 4. 并行主程序
# ==========================================
def run_parallel_experiments(json_file, target_n=20, time_limit=3600):
    print(f"Loading {json_file}...")
    with open(json_file, 'r') as f: all_instances = json.load(f)

    target_instances = [inst for inst in all_instances if inst['n'] == target_n]
    total_tasks = len(target_instances)

    num_workers, threads_per_job = 3, 2

    print(f"Running {total_tasks} instances (n={target_n}). Time Limit: {time_limit}s")
    print("-" * 130)
    print(
        f"{'Inst_ID':<8} {'Dist':<13} {'W_d':<5} {'Obj_G':<8} {'LB_Best':<8} {'Gap_FFD(%)':<12} {'Time(s)':<8} {'Stat':<6} {'Progress'}")
    print("-" * 130)

    tasks = [(inst, time_limit, threads_per_job) for inst in target_instances]
    results = []

    with Pool(processes=num_workers) as pool:
        for res in pool.imap_unordered(solve_instance_worker, tasks):
            results.append(res)

            # --- 进度追踪逻辑 ---
            completed = len(results)
            remaining = total_tasks - completed

            status_str = "OPT" if res["Status"] == 2 else "TIME"
            print(
                f"{res['Instance_ID']:<8} {res['Distribution'][:12]:<13} {res['W_d']:<5} {res['Obj_Gurobi']:<8} {res['LB_Best']:<8} {res['Gap_FFD_OS(%)']:<12} {res['Time(s)']:<8} {status_str:<6} | 还剩 {remaining} 个算例")

    results.sort(key=lambda x: (x['Distribution'], x['v'], x['W_d'], x['Instance_ID']))
    csv_file = f"results_n{target_n}_parallel_P3_base_{time_limit}s.csv"
    with open(csv_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f"\nAll done! Perfect CSV generated: {csv_file}")


if __name__ == "__main__":
    # 请根据需要修改目标文件、规模 target_n 和时长 time_limit
    run_parallel_experiments('base_instances_P3.json', target_n=200, time_limit=600)