import os
import numpy as np
from bs4 import BeautifulSoup

def safe_float(val, default_value=0.0):
    """安全气囊：遇到空数据不报错，自动填充默认值"""
    if val is None:
        return default_value
    val_str = str(val).strip()
    if val_str == '':
        return default_value
    try:
        return float(val_str)
    except ValueError:
        return default_value

def parse_6sigma_xml(file_path):
    """解析单个 6SigmaDC XML 文件"""
    with open(file_path, 'r', encoding='utf-8') as f:
        soup = BeautifulSoup(f.read(), 'xml')
    
    worksheets = soup.find_all('Worksheet')
    
    ac_supply_temps, ac_fan_speeds, ac_return_temps = [], [], []
    rack_powers, rack_outlet_temps = [], []
    
    for ws in worksheets:
        sheet_name = ws.get('ss:Name')
        rows = ws.find_all('Row')
        
        # 1. 解析【空调参数】表
        if sheet_name == '空调参数':
            for row in rows[1:]: # 跳过表头
                cells = row.find_all('Cell')
                if len(cells) >= 5:
                    name = cells[0].Data.text if cells[0].Data else ""
                    # 只抓取名字里带 "ACU" 的空调行，避开 "Room" 等统计行
                    if "ACU" in name:
                        # 如果没有数据，默认回风25度，送风20度，转速50%
                        ac_return_temps.append(safe_float(cells[2].Data.text, 25.0))
                        ac_supply_temps.append(safe_float(cells[3].Data.text, 20.0))
                        ac_fan_speeds.append(safe_float(cells[4].Data.text, 50.0))
                        
        # 2. 解析【机柜温度】表
        elif sheet_name == '机柜温度':
            for row in rows[1:]: # 跳过表头
                cells = row.find_all('Cell')
                if len(cells) >= 6:
                    name = cells[0].Data.text if cells[0].Data else ""
                    # 只抓取名字里带 "-" 的真实机柜行 (例如 "7-6", "1-1")
                    if "-" in name:
                        # 如果没开机导致没温度，默认 25.0度；功率默认为 0.0
                        rack_outlet_temps.append(safe_float(cells[4].Data.text, 25.0))
                        rack_powers.append(safe_float(cells[5].Data.text, 0.0))

    # 组装 X 和 Y
    X_single = ac_supply_temps + ac_fan_speeds + rack_powers
    Y_single = rack_outlet_temps + ac_return_temps
    
    return X_single, Y_single

def build_dataset(base_folder):
    X_list, Y_list = [], []
    success_count, fail_count = 0, 0
    
    print(f"开始扫描文件夹: {base_folder} ...")
    
    for root, dirs, files in os.walk(base_folder):
        for file in files:
            if file.endswith('.xml'):
                file_path = os.path.join(root, file)
                try:
                    X_single, Y_single = parse_6sigma_xml(file_path)
                    
                    # 严格校验：必须是 48 个输入，45 个输出
                    if len(X_single) == 48 and len(Y_single) == 45:
                        X_list.append(X_single)
                        Y_list.append(Y_single)
                        success_count += 1
                    else:
                        print(f"警告：文件 {file} 抓取的数量不对 (X:{len(X_single)}, Y:{len(Y_single)})，已跳过。")
                        fail_count += 1
                except Exception as e:
                    print(f"解析 {file} 彻底失败: {e}")
                    fail_count += 1
                    
    X_train = np.array(X_list, dtype=np.float32)
    Y_train = np.array(Y_list, dtype=np.float32)
    
    print("-" * 40)
    print(f"数据提取完美收官！成功解析: {success_count} 个文件。")
    print(f"特征矩阵 X_train 形状: {X_train.shape} (期望是 N x 48)")
    print(f"标签矩阵 Y_train 形状: {Y_train.shape} (期望是 N x 45)")
    print("-" * 40)
    
    return X_train, Y_train

if __name__ == "__main__":
    dataset_folder = r"E:\AA文件\研究生\DC-PI-GNN\data" 
    
    if not os.path.exists(dataset_folder):
        print(f"错误：找不到路径 {dataset_folder}，请检查盘符和文件夹名称！")
    else:
        X_train, Y_train = build_dataset(dataset_folder)
        
        np.save('X_train.npy', X_train)
        np.save('Y_train.npy', Y_train)
        print("大功告成！'X_train.npy' 和 'Y_train.npy' 已生成在当前代码目录下！你可以去运行 PI-GNN 训练代码了！")