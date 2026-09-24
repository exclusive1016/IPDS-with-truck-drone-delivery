import numpy as np
import pandas as pd
import math
import hashlib


def get_deterministic_seed(prob, n, dist, inst_id):
    """
    根据问题类型、规模、分布和实例ID生成绝对固定的随机数种子。
    确保不同环境下的可复现性。
    """
    seed_str = f"{prob}_{n}_{dist}_{inst_id}"
    return int(hashlib.md5(seed_str.encode('utf-8')).hexdigest(), 16) % (2 ** 32)


def generate_all_instances():
    # ==========================================
    # 1. 基础算例参数设定 (Base Testbed)
    # ==========================================
    problems = ['P3', 'P4']
    n_list = [20, 50, 100, 200]
    Wd_list = [0.15, 0.25, 0.35]
    v_list = [1.5, 2.0, 3.0]
    dist_list = ['Uniform', 'Right-skewed', 'Left-skewed']
    num_instances = 10

    all_instances = []

    print("开始生成基础算例 (Base Testbed)...")
    for prob in problems:
        for n in n_list:
            for dist in dist_list:
                for inst_id in range(1, num_instances + 1):

                    # 设定固定的随机数种子
                    seed = get_deterministic_seed(prob, n, dist, inst_id)
                    np.random.seed(seed)

                    # 生成 p_j
                    if prob == 'P3':
                        p_j = np.full(n, 15)
                    else:
                        p_j = np.random.randint(5, 26, size=n)

                    # 生成独立的 w_j
                    if dist == 'Uniform':
                        w_j = np.random.uniform(0.05, 0.5, size=n)
                    elif dist == 'Right-skewed':
                        w_j = np.random.beta(2, 5, size=n) * 0.45 + 0.05
                    elif dist == 'Left-skewed':
                        w_j = np.random.beta(5, 2, size=n) * 0.45 + 0.05

                    # 遍历车辆参数
                    for Wd in Wd_list:
                        for v in v_list:
                            T_d = 2 * math.ceil(np.sum(p_j) / n)
                            T_t = v * T_d

                            all_instances.append({
                                'Testbed': 'Base',
                                'Problem': prob,
                                'n': n,
                                'Distribution': dist,
                                'Wd': Wd,
                                'v': v,
                                'Instance_ID': inst_id,
                                'T_d': T_d,
                                'T_t': float(T_t),
                                'p_j': p_j.tolist(),
                                'w_j': np.round(w_j, 4).tolist()
                            })

    # ==========================================
    # 2. 相关性压力测试算例设定 (Correlated Testbed)
    # ==========================================
    print("开始生成相关性压力测试算例 (Correlated Testbed)...")
    correlated_Wd = 0.25
    correlated_v = 2.0

    for n in n_list:
        for inst_id in range(1, num_instances + 1):
            # 设定固定的随机数种子 (标识为 Correlated)
            seed = get_deterministic_seed('P4', n, 'Correlated', inst_id)
            np.random.seed(seed)

            # 生成 p_j (P4)
            p_j = np.random.randint(5, 26, size=n)

            # 引入 Uniform 噪声 epsilon
            epsilon = np.random.uniform(-0.02, 0.02, size=n)

            # 线性映射公式生成正相关 w_j
            w_j_raw = ((p_j - 5) / 20) * 0.45 + 0.05 + epsilon

            # 使用 np.clip 完美实现公式中的 min(0.5, max(0.05, ...)) 截断
            w_j = np.clip(w_j_raw, 0.05, 0.5)

            # 计算时间和周期
            T_d = 2 * math.ceil(np.sum(p_j) / n)
            T_t = correlated_v * T_d

            all_instances.append({
                'Testbed': 'Correlated',
                'Problem': 'P4',
                'n': n,
                'Distribution': 'Correlated',
                'Wd': correlated_Wd,
                'v': correlated_v,
                'Instance_ID': inst_id,
                'T_d': T_d,
                'T_t': float(T_t),
                'p_j': p_j.tolist(),
                'w_j': np.round(w_j, 4).tolist()
            })

    return pd.DataFrame(all_instances)


if __name__ == "__main__":
    dataset_df = generate_all_instances()

    # 验证生成数量
    print(f"数据生成完毕！总算例数: {len(dataset_df)} (预期 2200)")

    # 将数据拆分并保存
    p3_base = dataset_df[(dataset_df['Problem'] == 'P3') & (dataset_df['Testbed'] == 'Base')]
    p4_base = dataset_df[(dataset_df['Problem'] == 'P4') & (dataset_df['Testbed'] == 'Base')]
    p4_corr = dataset_df[dataset_df['Testbed'] == 'Correlated']

    p3_base.to_json("base_instances_P3.json", orient='records', indent=4)
    p4_base.to_json("base_instances_P4.json", orient='records', indent=4)
    p4_corr.to_json("correlated_instances_P4.json", orient='records', indent=4)

    print("数据已成功分类保存：")
    print(f"- P3 基础算例: {len(p3_base)} 个")
    print(f"- P4 基础算例: {len(p4_base)} 个")
    print(f"- P4 压力测试算例: {len(p4_corr)} 个")