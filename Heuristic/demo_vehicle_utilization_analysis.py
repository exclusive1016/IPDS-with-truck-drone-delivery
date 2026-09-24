import numpy as np
import pandas as pd
import time


# ==========================================
# 1. 基础打包与排序规则定义
# ==========================================
def FFD_packing(jobs, W_t=1.0):
    sorted_jobs = sorted(jobs, key=lambda x: x['w'], reverse=True)
    batches = []
    for job in sorted_jobs:
        placed = False
        for b in batches:
            b_weight = sum(j['w'] for j in b)
            if round(b_weight + job['w'], 4) <= W_t:
                b.append(job)
                placed = True
                break
        if not placed:
            batches.append([job])
    return batches


def NFD_packing(jobs, W_t=1.0):
    sorted_jobs = sorted(jobs, key=lambda x: x['w'], reverse=True)
    batches = []
    if not sorted_jobs:
        return batches
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


def calc_truck_load_util(batches, W_t=1.0):
    """计算卡车装载率 (Truck Load Utilization)"""
    if not batches:
        return 0.0
    total_weight = sum(sum(j['w'] for j in b) for b in batches)
    return (total_weight / (len(batches) * W_t)) * 100


# ==========================================
# 2. 核心调度程序 Procedure OS
# ==========================================
def Procedure_OS(truck_batches, drone_jobs, T_t, T_d):
    tb_info = [{'batch': b, 'P': sum(j['p'] for j in b)} for b in truck_batches]
    tb_info.sort(key=lambda x: x['P'])

    dj_info = [{'job': j, 'P': j['p']} for j in drone_jobs]
    dj_info.sort(key=lambda x: x['P'])

    b_count = len(tb_info)
    for i, tb in enumerate(tb_info):
        tb['q'] = (b_count - i) * T_t

    l_count = len(dj_info)
    for j, dj in enumerate(dj_info):
        dj['q'] = (l_count - j) * T_d

    all_tasks = tb_info + dj_info
    all_tasks.sort(key=lambda x: x['q'], reverse=True)

    current_C = 0
    makespan = 0
    for task in all_tasks:
        current_C += task['P']
        makespan = max(makespan, current_C + task['q'])

    return makespan


# ==========================================
# 3. 算法实现 (全面加入统计追踪)
# ==========================================
def Baseline_Truck_Only(jobs, T_t, T_d):
    truck_batches = FFD_packing(jobs)
    mksp = Procedure_OS(truck_batches, [], T_t, T_d)
    stats = {
        'Truck_Batches': len(truck_batches),
        'Truck_Util(%)': calc_truck_load_util(truck_batches)
    }
    return mksp, stats


def Baseline_Max_Drone(jobs, W_d, T_t, T_d):
    drone_jobs = [j for j in jobs if j['w'] <= W_d]
    truck_jobs = [j for j in jobs if j['w'] > W_d]
    truck_batches = FFD_packing(truck_jobs)
    mksp = Procedure_OS(truck_batches, drone_jobs, T_t, T_d)
    stats = {
        'Truck_Batches': len(truck_batches),
        'Truck_Util(%)': calc_truck_load_util(truck_batches)
    }
    return mksp, stats


def Algorithm_Proposed_OS(jobs, W_d, T_t, T_d, pack_rule='FFD'):
    J_t = [j for j in jobs if j['w'] > W_d]
    J_d_cand = [j for j in jobs if j['w'] <= W_d]
    J_d_cand.sort(key=lambda x: x['w'], reverse=True)

    best_makespan = float('inf')
    best_stats = {}
    l_range = range(0, len(J_d_cand) + 1)
    # l_range = range(1, len(J_d_cand) + 1) if len(J_d_cand) > 0 else [0]

    for l in l_range:
        drone_jobs = J_d_cand[:l]
        truck_jobs = J_t + J_d_cand[l:]

        truck_batches = FFD_packing(truck_jobs) if pack_rule == 'FFD' else NFD_packing(truck_jobs)
        mksp = Procedure_OS(truck_batches, drone_jobs, T_t, T_d)

        if mksp < best_makespan:
            best_makespan = mksp
            best_stats = {
                'Used_Num': len(drone_jobs),  # 用于计算 Assignment Rate
                'Truck_Batches': len(truck_batches),
                'Truck_Util(%)': calc_truck_load_util(truck_batches)
            }
    return best_makespan, best_stats


# ==========================================
# 4. 实验运行与结果整理
# ==========================================
def run_experiments_for_problem(json_file, problem_type):
    print(f"-> 正在处理 {problem_type} 数据集: {json_file}")
    df = pd.read_json(json_file)
    results = []

    for index, row in df.iterrows():
        n = row['n']
        jobs = [{'id': i, 'p': row['p_j'][i], 'w': row['w_j'][i]} for i in range(n)]
        W_d, T_t, T_d = row['Wd'], row['T_t'], row['T_d']

        # 核心物理基准：符合条件的无人机工件数量
        eligible_num = len([j for j in jobs if j['w'] <= W_d])
        safe_eligible = eligible_num if eligible_num > 0 else 1  # 防除以0

        record = {
            'n': n,
            'W_d': W_d,
            'v': row['v'],
            'Distribution': row.get('Weight_Distribution', row.get('Distribution')),
            'Instance_ID': row['Instance_ID'],
            'Drone-Eligible_Jobs(%)': (eligible_num / n) * 100
        }

        # --- 运行所有算法 ---
        start = time.perf_counter()
        mksp_t, stat_t = Baseline_Truck_Only(jobs, T_t, T_d)
        record['Time_Truck(s)'] = time.perf_counter() - start

        start = time.perf_counter()
        mksp_m, stat_m = Baseline_Max_Drone(jobs, W_d, T_t, T_d)
        record['Time_MaxDrone(s)'] = time.perf_counter() - start

        start = time.perf_counter()
        mksp_ffd, stat_ffd = Algorithm_Proposed_OS(jobs, W_d, T_t, T_d, pack_rule='FFD')
        record['Time_FFD_OS(s)'] = time.perf_counter() - start

        if problem_type == 'P3':
            start = time.perf_counter()
            mksp_nfd, stat_nfd = Algorithm_Proposed_OS(jobs, W_d, T_t, T_d, pack_rule='NFD')
            record['Time_NFD_OS(s)'] = time.perf_counter() - start

        # --- 计算 Best 最优标志 (非常适合求和统计胜率) ---
        min_mksp = min([mksp_t, mksp_m, mksp_ffd] + ([mksp_nfd] if problem_type == 'P3' else []))
        record['Best_Truck'] = 1 if abs(mksp_t - min_mksp) < 1e-5 else 0
        record['Best_MaxDrone'] = 1 if abs(mksp_m - min_mksp) < 1e-5 else 0
        record['Best_FFD_OS'] = 1 if abs(mksp_ffd - min_mksp) < 1e-5 else 0
        if problem_type == 'P3':
            record['Best_NFD_OS'] = 1 if abs(mksp_nfd - min_mksp) < 1e-5 else 0

        # --- 填入新指标 ---
        record.update({
            'Mksp_Truck': mksp_t,
            'Truck_Batches(TruckOnly)': stat_t['Truck_Batches'],
            'Truck_Load_Util_TruckOnly(%)': stat_t['Truck_Util(%)'],

            'Mksp_MaxDrone': mksp_m,
            'Truck_Batches(MaxDrone)': stat_m['Truck_Batches'],
            'Truck_Load_Util_MaxDrone(%)': stat_m['Truck_Util(%)'],

            'Mksp_FFD_OS': mksp_ffd,
            'FFD_Drone_Assignment_Rate(%)': (stat_ffd['Used_Num'] / safe_eligible) * 100 if eligible_num > 0 else 0.0,
            'Truck_Batches(FFD)': stat_ffd['Truck_Batches'],
            'Truck_Load_Util_FFD(%)': stat_ffd['Truck_Util(%)']
        })

        if problem_type == 'P3':
            record.update({
                'Mksp_NFD_OS': mksp_nfd,
                'NFD_Drone_Assignment_Rate(%)': (stat_nfd[
                                                     'Used_Num'] / safe_eligible) * 100 if eligible_num > 0 else 0.0,
                'Truck_Batches(NFD)': stat_nfd['Truck_Batches'],
                'Truck_Load_Util_NFD(%)': stat_nfd['Truck_Util(%)']
            })

        # --- 保留原有的 Gap 记录，以备在附录中使用 ---
        record['Min_Makespan_Heuristic'] = min_mksp
        record['Gap_Truck(%)'] = (mksp_t - min_mksp) / min_mksp * 100
        record['Gap_MaxDrone(%)'] = (mksp_m - min_mksp) / min_mksp * 100
        record['Gap_FFD_OS(%)'] = (mksp_ffd - min_mksp) / min_mksp * 100
        if problem_type == 'P3':
            record['Gap_NFD_OS(%)'] = (mksp_nfd - min_mksp) / min_mksp * 100

        results.append(record)

    return pd.DataFrame(results)


if __name__ == "__main__":
    print("开始执行完备实验计算...")

    df_p3_base = run_experiments_for_problem("base_instances_P3.json", 'P3')
    df_p4_base = run_experiments_for_problem("base_instances_P4.json", 'P4')

    output_filename = 'Simulation_Results_with_Detailed_Stats.xlsx'
    with pd.ExcelWriter(output_filename) as writer:
        df_p4_base.to_excel(writer, sheet_name='P4_Base_Results', index=False)
        df_p3_base.to_excel(writer, sheet_name='P3_Base_Results', index=False)

    print(f"✅ 写入成功！所有新指标已保存至：{output_filename}")