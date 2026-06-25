import pandas as pd
import numpy as np
import os
from scipy.optimize import differential_evolution
from predict import run_prediction

# ==========================================
# 1. 基础配置
# ==========================================
TARGET_DATE = "2025-06-21"
EXCEL_FILE = "数据中心数据清单0901.xlsx"

# 物理与安全常量
SAFE_OUTLET_LIMIT = 40.0  # 机柜出风红线
P_UPS_LOSS_N = 2.5       # 北区 UPS 损耗
CARBON_FACTOR = 0.581    # 碳排放因子 (kg CO2/kWh)

# 检查文件是否存在
if not os.path.exists(EXCEL_FILE):
    print(f"❌ 错误：在当前目录下找不到 {EXCEL_FILE}")
    exit()

# ==========================================
# 2. 自动从 Excel 读取数据
# ==========================================
def get_daily_context(date_str):
    print(f"📊 正在从 Excel 读取 {date_str} 的数据...")
    
    # --- A. 读取机柜功率分布 ---
    df_r = pd.read_excel(EXCEL_FILE, sheet_name="机柜")
    df_r['时间'] = df_r['时间'].astype(str)
    
    day_r_mask = df_r['时间'].str.contains(date_str)
    if not day_r_mask.any():
        print(f"❌ 错误：在'机柜'表中找不到日期 {date_str}")
        exit()
    day_r = df_r[day_r_mask].iloc[0]
    
    rack_powers = []
    for col in range(1, 8):
        for r in range(1, 7):
            col_name = f"N-{col}_{r}"
            if col_name in day_r:
                rack_powers.append(float(day_r[col_name]))
            else:
                rack_powers.append(0.0)
    
    # --- B. 读取原始冷却功耗 ---
    df_c = pd.read_excel(EXCEL_FILE, sheet_name="冷却、照明系统")
    
    # 清理列名首尾空格
    df_c.columns = [str(c).strip() for c in df_c.columns]
    
    time_col = '时间' if '时间' in df_c.columns else '日期'
    df_c[time_col] = df_c[time_col].astype(str)
    
    day_c_mask = df_c[time_col].str.contains(date_str)
    if not day_c_mask.any():
        print(f"❌ 错误：在'冷却'表中找不到日期 {date_str}")
        exit()
    day_c = df_c[day_c_mask].iloc[0]
    
    # 读取 5 台空调总功率，直接乘以 3/5
    col_name = '精密空调消耗功率（kw）'
    if col_name in day_c:
        total_ac_val = pd.to_numeric(day_c[col_name], errors='coerce')
        total_ac_power = float(total_ac_val) if not pd.isna(total_ac_val) else 0.0
    else:
        # 保底读取第二列
        total_ac_val = pd.to_numeric(day_c.iloc[1], errors='coerce')
        total_ac_power = float(total_ac_val) if not pd.isna(total_ac_val) else 0.0
        
    orig_cooling = total_ac_power * (3.0 / 5.0)
    
    print(f"✅ 数据加载成功：IT总功耗={sum(rack_powers):.2f}kW")
    print(f"❄️ 5台空调总功耗={total_ac_power:.2f}kW -> 北机房(3/5)={orig_cooling:.2f}kW")
    
    return sum(rack_powers), rack_powers, orig_cooling

# ==========================================
# 3. 寻优核心逻辑
# ==========================================
def calculate_north_cooling_power(v_ac1, v_ac2, t_ac1, t_ac2):
    fan_p = 3.0 * ((v_ac1 / 100.0)**3) + 3.0 * ((v_ac2 / 100.0)**3)
    comp_p = 12.0 * (1 + 0.05 * (24.0 - t_ac1)) + 12.0 * (1 + 0.05 * (24.0 - t_ac2))
    ups_ac_p = 6.0 
    return fan_p + comp_p + ups_ac_p

def objective_function(x, p_it, rack_powers):
    v_ac1, v_ac2, t_ac1, t_ac2 = x
    p_cool = calculate_north_cooling_power(v_ac1, v_ac2, t_ac1, t_ac2)
    max_t = run_prediction(opt_params=(v_ac1, v_ac2, t_ac1, t_ac2), rack_powers=rack_powers)
    
    penalty = 0.0
    if max_t > SAFE_OUTLET_LIMIT:
        penalty = 20000.0 * (max_t - SAFE_OUTLET_LIMIT)**2
        
    return (p_it + p_cool + P_UPS_LOSS_N) + penalty

if __name__ == "__main__":
    p_it_val, rack_list, p_cool_orig = get_daily_context(TARGET_DATE)
    
    print(f"📊 实时 IT 总负载: {p_it_val:.2f} kW | 北区原始冷却功耗: {p_cool_orig:.2f} kW")
    
    bounds = [(40.0, 100.0), (40.0, 100.0), (18.0, 27.0), (18.0, 27.0)]
    
    result = differential_evolution(
        objective_function, 
        bounds, 
        args=(p_it_val, rack_list),
        strategy='best1bin', 
        maxiter=30, 
        popsize=10, 
        disp=True
    )
    
    # --- 结果分析与计算 ---
    v1, v2, t1, t2 = result.x
    opt_cool = calculate_north_cooling_power(v1, v2, t1, t2)
    
    orig_pue = (p_it_val + p_cool_orig + P_UPS_LOSS_N) / p_it_val
    opt_pue = (p_it_val + opt_cool + P_UPS_LOSS_N) / p_it_val
    
    # 减碳与节电计算 (假设全天 24 小时保持此状态)
    power_saved_kw = p_cool_orig - opt_cool            # 节省的功率 (kW)
    energy_saved_kwh = power_saved_kw * 24.0           # 日节电量 (kWh)
    carbon_reduced_kg = energy_saved_kwh * CARBON_FACTOR # 日减碳量 (kg)
    
    print("\n" + "★"*50)
    print(f"🏆 {TARGET_DATE} 寻优任务完成")
    print(f"▶ 原始 PUE (仅北区): {orig_pue:.3f} | 优化后 PUE: {opt_pue:.3f}")
    print(f"▶ 能效提升比例: {((orig_pue - opt_pue)/orig_pue)*100:.2f}%")
    print(f"▶ 日节电量: {energy_saved_kwh:.1f} kWh | 🌍 日减碳量: {carbon_reduced_kg:.1f} kg CO2")
    print(f"▶ 推荐策略: AC1({v1:.1f}%, {t1:.1f}℃) | AC2({v2:.1f}%, {t2:.1f}℃)")
    print("★"*50)
    # ==========================================
    # 【新增】：将寻优结果自动追加保存到 txt 文件中
    # ==========================================
    log_filename = "Optimization_Log_NorthRoom.txt"
    
    # 使用 "a" (append) 模式，这样每次跑不同的日期都会往下追加，不会覆盖以前的记录
    with open(log_filename, "a", encoding="utf-8") as f:
        f.write(f"★" * 50 + "\n")
        f.write(f"🏆 日期: {TARGET_DATE} | 寻优任务完成\n")
        f.write(f"▶ 原始 PUE (仅北区): {orig_pue:.3f} | 优化后 PUE: {opt_pue:.3f}\n")
        f.write(f"▶ 能效提升比例: {((orig_pue - opt_pue)/orig_pue)*100:.2f}%\n")
        f.write(f"▶ 日节电量: {energy_saved_kwh:.1f} kWh | 🌍 日减碳量: {carbon_reduced_kg:.1f} kg CO2\n")
        f.write(f"▶ 推荐策略: AC1({v1:.1f}%, {t1:.1f}℃) | AC2({v2:.1f}%, {t2:.1f}℃)\n")
        f.write(f"★" * 50 + "\n\n")
        
    print(f"\n📄 寻优结果已自动保存至日志文件: {log_filename}")