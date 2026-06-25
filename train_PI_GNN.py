import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.nn import GATConv
from torch_geometric.data import Data, DataLoader
import networkx as nx
import math
import numpy as np

# ==========================================
# 模块一：高保真三维物理拓扑建图
# ==========================================
def build_digital_twin_graph():
    G = nx.DiGraph()
    pos = {} 
    
    # 1. 确定机柜 X 坐标 (东西向，由东向西分别为 列1 到 列7)
    col_x_coords = {
        7: 1.7, 6: 3.9, 5: 6.1, 4: 8.3, 
        3: 10.5, 2: 12.7, 1: 14.9
    }

    # 2. 确定机柜 Y 坐标 (南北向，北侧为Y=0有空调，南侧Y增大)
    row_y_coords = {}
    for row in range(1, 7):
        row_y_coords[row] = 0.6 + (6 - row) * 0.8 + 0.4 

    # 3. 添加空调节点 (北墙 Y=0)
    G.add_node(42, type='AC')
    pos[42] = (5.0, 0.0) # AC1(西) 
    G.add_node(43, type='AC')
    pos[43] = (12.7, 0.0) # AC2(东) 
    
    # 4. 添加 42 个机柜节点，并记录高开孔率(45%)的特殊机柜位置
    high_porosity_racks = [(3, 3), (3, 6), (6, 2), (6, 5)] 
    idx = 0
    for col in range(1, 8):
        for row in range(1, 7):
            G.add_node(idx, type='Rack', col=col, row=row)
            pos[idx] = (col_x_coords[col], row_y_coords[row])
            idx += 1

    # 5. 建立包含真实物理特征的边
    for idx in range(42):
        col = G.nodes[idx]['col']
        row = G.nodes[idx]['row']
        rack_pos = pos[idx]
        
        # 计算静压箱内实际送风路径直线距离 (米)
        dist_ac1 = math.sqrt((pos[42][0] - rack_pos[0])**2 + (pos[42][1] - rack_pos[1])**2)
        dist_ac2 = math.sqrt((pos[43][0] - rack_pos[0])**2 + (pos[43][1] - rack_pos[1])**2)
        
        porosity = 0.45 if (col, row) in high_porosity_racks else 0.22
            
        # 西侧 (列4,5,6,7) 受 AC1 控制；东侧 (列1,2,3) 受 AC2 控制
        if col >= 4: 
            G.add_edge(42, idx, distance=dist_ac1, porosity=porosity)
            G.add_edge(idx, 42, distance=dist_ac1, porosity=0.0)
        else:
            G.add_edge(43, idx, distance=dist_ac2, porosity=porosity)
            G.add_edge(idx, 43, distance=dist_ac2, porosity=0.0)
            
        # 机柜间的热传导边 (南北向相邻机柜距离 0.8m)
        if row < 6:
            idx_north = idx + 1 
            G.add_edge(idx, idx_north, distance=0.8, porosity=0.0)
            G.add_edge(idx_north, idx, distance=0.8, porosity=0.0)

    # 提取 PyG 格式数据
    edges = list(G.edges(data=True))
    edge_index = torch.tensor([[e[0] for e in edges], [e[1] for e in edges]], dtype=torch.long)
    edge_attr = torch.tensor([[e[2]['distance'], e[2]['porosity']] for e in edges], dtype=torch.float)
    
    return edge_index, edge_attr

# ==========================================
# 模块二：双头物理图注意力网络 (Dual-Head PI-GAT)
# ==========================================
class AdvancedPhysicsInformedGAT(nn.Module):
    def __init__(self, node_in_dim, edge_in_dim, hidden_dim=64):
        super(AdvancedPhysicsInformedGAT, self).__init__()
        self.conv1 = GATConv(node_in_dim, hidden_dim, edge_dim=edge_in_dim, heads=4, concat=False)
        self.conv2 = GATConv(hidden_dim, hidden_dim, edge_dim=edge_in_dim, heads=4, concat=False)
        
        # 预测：[0列: 出风温度, 1列: 局部风量]
        self.fc_out = nn.Linear(hidden_dim, 2) 

    def forward(self, x, edge_index, edge_attr):
        x = F.mish(self.conv1(x, edge_index, edge_attr))
        x = F.mish(self.conv2(x, edge_index, edge_attr))
        outputs = self.fc_out(x)
        return outputs

# ==========================================
# 模块三：精细化局部物理定律损失函数
# ==========================================
def compute_fine_grained_pignn_loss(model_outputs, targets_T, node_features, lambda_data=1.0):
    pred_T_out = model_outputs[:, 0].unsqueeze(1)    # 预测温度
    pred_V_local = model_outputs[:, 1].unsqueeze(1)  # 预测分配风量
    
    # 1. 传统数据驱动 Loss (MSE)
    mse_loss_T = F.mse_loss(pred_T_out, targets_T)
    
    # 提取物理常量与节点特征
    Cp = 1.005        
    rho = 1.19        
    V_max = 10.0      
    
    rack_powers = node_features[0:42, 0] 
    T_supply = node_features[42, 2] # 取 AC1 的送风温度作为基准
    
    # 2. 物理定律 A：局部能量守恒
    calculated_heat = Cp * rho * pred_V_local[0:42, 0] * (pred_T_out[0:42, 0] - T_supply)
    loss_local_energy = torch.mean((rack_powers - calculated_heat) ** 2)
    
    # 3. 物理定律 B：质量/风量守恒
    total_V_predicted = torch.sum(pred_V_local[0:42, 0])
    actual_total_V = (V_max * (node_features[42, 1] / 100.0)) + (V_max * (node_features[43, 1] / 100.0))
    loss_mass = (total_V_predicted - actual_total_V) ** 2
    
    # 4. 物理定律 C：热力学常识越界惩罚
    loss_thermo_T = torch.mean(F.relu(T_supply - pred_T_out[0:42, 0])) 
    loss_thermo_V = torch.mean(F.relu(-pred_V_local[0:42, 0]))

    # 综合物理损失
    physics_loss = (0.01 * loss_local_energy) + (0.1 * loss_mass) + (1.0 * loss_thermo_T) + (1.0 * loss_thermo_V)
    total_loss = lambda_data * mse_loss_T + physics_loss
    
    return total_loss, mse_loss_T, physics_loss

# ==========================================
# 模块四：数据转换器 (Numpy -> PyG Graphs)
# ==========================================
def prepare_graph_dataset(X_numpy, Y_numpy, edge_index, edge_attr):
    """
    将扁平的 Numpy 数据转化为图神经网络可识别的 44 个节点特征矩阵
    """
    dataset = []
    num_samples = X_numpy.shape[0]
    
    for i in range(num_samples):
        # 初始化节点特征 [44节点, 3特征(功率, 风机转速, 送风温度)]
        node_features = torch.zeros((44, 3), dtype=torch.float)
        
        # 1. 填充 42个机柜特征 (X的6~47列是机柜功率)
        # 向量赋值 PyTorch 可以自动兼容
        node_features[0:42, 0] = torch.tensor(X_numpy[i, 6:48], dtype=torch.float)
        
        # 2. 填充 2台主空调特征 (AC1=节点42, AC2=节点43)
        # 【修复点】：加上 .item() 将 numpy.float32 强转为 python 原生 float
        # 空调1：风机转速在X[3], 送风温度在X[0]
        node_features[42, 1] = X_numpy[i, 3].item() 
        node_features[42, 2] = X_numpy[i, 0].item() 
        # 空调2：风机转速在X[4], 送风温度在X[1]
        node_features[43, 1] = X_numpy[i, 4].item() 
        node_features[43, 2] = X_numpy[i, 1].item() 
        
        # 3. 初始化目标标签 [44节点, 1特征(真实温度)]
        targets = torch.zeros((44, 1), dtype=torch.float)
        # 机柜的出风温度 (Y的0~41列)
        targets[0:42, 0] = torch.tensor(Y_numpy[i, 0:42], dtype=torch.float)
        
        # 【修复点】：空调的回风温度同样加上 .item()
        targets[42, 0] = Y_numpy[i, 42].item()
        targets[43, 0] = Y_numpy[i, 43].item()
        
        # 封装为 PyG 数据对象
        data = Data(x=node_features, edge_index=edge_index, edge_attr=edge_attr, y=targets)
        dataset.append(data)
        
    return dataset

from torch.utils.data import random_split
import os

import os

# ==========================================
# 模块五：正式训练循环 (带防过拟合验证)
# ==========================================
def train_pi_gnn():
    if not os.path.exists('X_train.npy') or not os.path.exists('Y_train.npy'):
        print("【错误】未找到数据文件！")
        return
        
    print("1. 正在加载真实仿真数据...")
    X_train_np = np.load('X_train.npy')
    Y_train_np = np.load('Y_train.npy')
    
    print("2. 正在构建机房图拓扑结构...")
    edge_index, edge_attr = build_digital_twin_graph()
    
    print("3. 数据切分与打包...")
    dataset = prepare_graph_dataset(X_train_np, Y_train_np, edge_index, edge_attr)
    
    # ================= 核心升级：划分训练集与测试集 =================
    total_size = len(dataset)
    train_size = int(0.9 * total_size) # 90% 用来训练 (240组)
    test_size = total_size - train_size # 10% 用来考试 (60组)
    
    # 随机打乱并切分
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False) # 考试不需要打乱
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = AdvancedPhysicsInformedGAT(node_in_dim=3, edge_in_dim=2).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # 增加轮数，让子弹再飞一会儿！
    epochs = 5000 
    print(f"\n================ 开始正式训练 (使用 {device}) ================")
    
    for epoch in range(epochs):
        # ------------------ 训练阶段 ------------------
        model.train()
        total_train_mse = 0
        
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            outputs = model(batch.x, batch.edge_index, batch.edge_attr)
            
            # 由于 batch 图合并，手动切分计算 Loss
            loss = 0; mse = 0; phys = 0
            num_graphs = batch.num_graphs
            for i in range(num_graphs):
                start_idx, end_idx = i * 44, (i + 1) * 44
                l, m, p = compute_fine_grained_pignn_loss(
                    outputs[start_idx:end_idx], batch.y[start_idx:end_idx], batch.x[start_idx:end_idx]
                )
                loss += l; mse += m; phys += p
                
            (loss / num_graphs).backward()
            optimizer.step()
            total_train_mse += (mse / num_graphs).item()
            
        avg_train_mse = total_train_mse / len(train_loader)
        
        # ------------------ 考试(验证)阶段 ------------------
        model.eval() # 关闭梯度计算，进入考试模式
        total_test_mse = 0
        total_test_phys = 0
        
        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(device)
                outputs = model(batch.x, batch.edge_index, batch.edge_attr)
                
                mse = 0; phys = 0
                num_graphs = batch.num_graphs
                for i in range(num_graphs):
                    start_idx, end_idx = i * 44, (i + 1) * 44
                    l, m, p = compute_fine_grained_pignn_loss(
                        outputs[start_idx:end_idx], batch.y[start_idx:end_idx], batch.x[start_idx:end_idx]
                    )
                    mse += m; phys += p
                    
                total_test_mse += (mse / num_graphs).item()
                total_test_phys += (phys / num_graphs).item()
                
        avg_test_mse = total_test_mse / len(test_loader)
        avg_test_phys = total_test_phys / len(test_loader)
        
        # 打印日志（对比训练和考试成绩）
        if (epoch + 1) % 50 == 0:
            print(f"Epoch [{epoch+1:04d}/{epochs}] | 训练集MSE: {avg_train_mse:.4f} | ⬇️ 测试集MSE(考试成绩): {avg_test_mse:.4f} | 测试集物理惩罚: {avg_test_phys:.4f}")

    print("\n================ 训练圆满结束 ================")
    torch.save(model.state_dict(), 'DataCenter_PIGNN_Final.pth')
    print("模型权重已保存！")

if __name__ == "__main__":
    train_pi_gnn()