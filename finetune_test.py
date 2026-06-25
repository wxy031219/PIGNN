import time
import numpy as np
from stable_baselines3 import PPO
from rl_env import DataCenterEnv

if __name__ == "__main__":
    env = DataCenterEnv()
    old_model_name = "ppo_datacenter_agent_V2_FullRacks"
    new_model_name = "ppo_datacenter_agent_V3_Finetuned"
    
    print(f"📥 正在加载已有的基础模型: {old_model_name}...")
    try:
        # 加载旧模型，并继续放在 GPU 上
        model = PPO.load(old_model_name, env=env, device='cuda')
    except Exception as e:
        print(f"❌ 加载失败: {e}")
        exit()

    # 只需要微调 50,000 步 (大概只需要 2-3 个小时)
    FINETUNE_STEPS = 50000
    print(f"🚀 开始微调，纠正局部最优策略，目标步数: {FINETUNE_STEPS}...")
    model.learn(total_timesteps=FINETUNE_STEPS)
    
    model.save(new_model_name)
    print(f"✅ 微调完成！已保存为 {new_model_name}.zip\n")

    # ==========================================
    # 修复后的在线实时调度测试
    # ==========================================
    print("★" * 50)
    print("⚡ 修复版：模拟突发动态任务在线调度")
    print("★" * 50)
    
    eval_model = PPO.load(new_model_name)
    test_loads = [42.5, 55.0, 68.5]
    
    for load in test_loads:
        # 【关键修复！】强制同步底层沙盒的物理发热量
        env.current_it_power = load  
        obs = np.array([load], dtype=np.float32)
        
        start_time = time.time()
        action, _states = eval_model.predict(obs, deterministic=True)
        end_time = time.time()
        
        _, reward, _, _, info = env.step(action)
        
        v1, v2 = np.interp(action[0:2], [-1, 1], [40, 100])
        t1, t2 = np.interp(action[2:4], [-1, 1], [18, 27])
        
        print(f"▶ [工况] 突发 IT 总负载: {load:.1f} kW")
        print(f"   ⏱️ 决策耗时: {(end_time - start_time)*1000:.2f} 毫秒")
        print(f"   ❄️ 输出策略: AC1({v1:.1f}%, {t1:.1f}℃) | AC2({v2:.1f}%, {t2:.1f}℃)")
        print(f"   🌡️ PI-GNN验证最高温: {info['Max_Temp']:.2f}℃")
        print(f"   ⚡ 最终系统 PUE: {info['PUE']:.3f}")
        print("-" * 50)