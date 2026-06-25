import pandas as pd
from bs4 import BeautifulSoup
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

# 设置画图的中文字体，防止乱码
matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # Windows默认黑体
matplotlib.rcParams['axes.unicode_minus'] = False

def parse_6sigma_truth(xml_file):
    print(f"正在解析 6SigmaDC 真实数据文件: {xml_file} ...")
    with open(xml_file, 'r', encoding='utf-8') as f:
        soup = BeautifulSoup(f.read(), 'xml')
        
    truth_dict = {}
    worksheets = soup.find_all('Worksheet')
    for ws in worksheets:
        # 1. 解析机柜真实温度
        if ws.get('ss:Name') == '机柜温度':
            for row in ws.find_all('Row')[1:]:
                cells = row.find_all('Cell')
                if len(cells) >= 5 and cells[0].Data is not None:
                    rack_id = cells[0].Data.text.strip()
                    if '-' in rack_id:
                        try:
                            true_temp = float(cells[4].Data.text)
                            truth_dict[f"机柜 {rack_id}"] = true_temp
                        except ValueError:
                            pass
                            
        # 2. 解析空调真实回风温度
        elif ws.get('ss:Name') == '空调参数':
            for row in ws.find_all('Row')[1:]:
                cells = row.find_all('Cell')
                if len(cells) >= 3 and cells[0].Data is not None:
                    ac_name = cells[0].Data.text.strip().lower()
                    try:
                        return_temp = float(cells[2].Data.text)
                        if 'acu01' in ac_name:
                            truth_dict['精密空调-AC1(西)'] = return_temp
                        elif 'acu02' in ac_name:
                            truth_dict['精密空调-AC2(东)'] = return_temp
                    except ValueError:
                        pass
    return truth_dict

def analyze_and_plot():
    xml_file = "导出对比数据.xml"
    csv_file = "Prediction_Results.csv"
    
    try:
        truth_dict = parse_6sigma_truth(xml_file)
        df_pred = pd.read_csv(csv_file)
    except Exception as e:
        print(f"【错误】读取文件失败: {e}")
        return

    results = []
    for index, row in df_pred.iterrows():
        device_name = row['设备编号']
        if device_name in truth_dict:
            pred_temp = float(row['预测_出风/回风温度(℃)'])
            true_temp = truth_dict[device_name]
            abs_error = abs(pred_temp - true_temp)
            
            results.append({
                '设备类型': row['设备类型'],
                '设备编号': device_name,
                '真实温度(℃)': true_temp,
                '预测温度(℃)': pred_temp,
                '绝对误差(℃)': abs_error
            })
            
    df_compare = pd.DataFrame(results)
    
    # 计算核心误差指标
    mae = df_compare['绝对误差(℃)'].mean()
    rmse = np.sqrt(((df_compare['预测温度(℃)'] - df_compare['真实温度(℃)']) ** 2).mean())
    max_error = df_compare['绝对误差(℃)'].max()
    
    print("\n================= 全局误差分析报告 =================")
    print(f"参与对比的设备总数: {len(df_compare)} (含空调)")
    print(f"平均绝对误差 (MAE) : {mae:.4f} ℃")
    print(f"均方根误差 (RMSE)  : {rmse:.4f} ℃")
    print(f"最大绝对误差       : {max_error:.4f} ℃")
    print("====================================================")
    
    print("\n⚠️ 误差最大的前 3 名设备 (请关注校准后的表现)：")
    worst_3 = df_compare.sort_values(by='绝对误差(℃)', ascending=False).head(3)
    for i, (_, row) in enumerate(worst_3.iterrows(), 1):
        print(f"Top {i}: {row['设备编号']} | 真实:{row['真实温度(℃)']:.2f}℃ | 预测:{row['预测温度(℃)']:.2f}℃ | 误差: {row['绝对误差(℃)']:.2f}℃")
    
    # 保存详细误差表
    df_compare.to_csv("Error_Analysis_Report_with_AC.csv", index=False, encoding='utf-8-sig')
    
    # ==========================================
    # 画图部分
    # ==========================================
    plt.figure(figsize=(18, 6))
    x = np.arange(len(df_compare))
    width = 0.35
    
    plt.bar(x - width/2, df_compare['真实温度(℃)'], width, label='真实温度 (6SigmaDC)', color='#1f77b4', alpha=0.85)
    plt.bar(x + width/2, df_compare['预测温度(℃)'], width, label='预测温度 (PI-GNN)', color='#ff7f0e', alpha=0.85)
    
    plt.ylabel('温度 (℃)', fontsize=12)
    plt.title('机柜与空调温度预测结果与 CFD 真实值对比 (加入推理期校准)', fontsize=16)
    plt.xticks(x, df_compare['设备编号'], rotation=45, ha='right', fontsize=9)
    
    # 标出误差大于 5.0 度的红星（测试看这次红星是否消失了）
    for i, row in df_compare.iterrows():
        if row['绝对误差(℃)'] > 5.0:
            plt.text(i, max(row['真实温度(℃)'], row['预测温度(℃)']) + 1, '★', color='red', ha='center', fontsize=14)

    plt.legend(fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig('Temperature_Comparison_with_AC.png', dpi=300)
    print("\n✅ 对比图片已保存为：Temperature_Comparison_with_AC.png")
    plt.show()

if __name__ == "__main__":
    analyze_and_plot()