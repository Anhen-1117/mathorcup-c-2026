import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

input_path = r'D:\数学建模\C题\附件1：样例数据.xlsx'
output_dir = r'D:\数学建模\C题'

df = pd.read_excel(input_path)
print(f"数据形状：{df.shape}")
print(f"列名：{df.columns.tolist()}")

df_phlegm = df[df['体质标签'] == 5].copy()
print(f"\n痰湿体质患者数量：{len(df_phlegm)}")

tcm_costs = {1: 30, 2: 80, 3: 130}
act_costs = {1: 3, 2: 5, 3: 8}
tcm_bonus = {1: 0, 2: 0.05, 3: 0.10}
tcm_names = {1: '基础调理', 2: '中度调理', 3: '强化调理'}
act_names = {1: '1 级强度', 2: '2 级强度', 3: '3 级强度'}
act_times = {1: '10 分钟', 2: '20 分钟', 3: '30 分钟'}

def get_tcm_level_range(phlegm_score):
    if phlegm_score <= 58:
        return [1]
    elif phlegm_score <= 61:
        return [1, 2]
    return [1, 2, 3]

def get_activity_level_range(age_group, activity_score):
    age_range = [1, 2, 3] if age_group in [1, 2] else ([1, 2] if age_group in [3, 4] else [1])
    act_range = [1] if activity_score < 40 else ([1, 2] if activity_score < 60 else [1, 2, 3])
    return list(set(age_range) & set(act_range))

def optimize_plan(phlegm_score, age_group, activity_score):
    tcm_range = get_tcm_level_range(phlegm_score)
    act_range = get_activity_level_range(age_group, activity_score)
    best_plan, best_score = None, -float('inf')
    
    for tcm in tcm_range:
        for act in act_range:
            for freq in range(1, 11):
                cost = tcm_costs[tcm] * 6 + act_costs[act] * freq * 24
                if cost > 2000:
                    continue
                monthly_rate = 0.03 * act + 0.01 * max(0, freq - 5)
                effect = min(0.60, (1 - (1 - monthly_rate) ** 6) + tcm_bonus[tcm])
                score = 0.7 * effect + 0.3 * (1 - cost / 2000)
                if score > best_score:
                    best_score = score
                    best_plan = {'tcm_level': tcm, 'activity_level': act, 'frequency': freq, 'cost': cost, 'effect_rate': effect}
    return best_plan

results = []
for idx, row in df_phlegm.iterrows():
    plan = optimize_plan(row['痰湿质'], row['年龄组'], row['活动量表总分（ADL总分+IADL总分）'])
    if plan:
        results.append({
            '样本ID': row['样本ID'],
            '痰湿质积分': row['痰湿质'],
            '年龄组': row['年龄组'],
            '活动量表总分': row['活动量表总分（ADL总分+IADL总分）'],
            '高血脂症': row['高血脂症二分类标签'],
            '中医调理等级': plan['tcm_level'],
            '活动干预强度': plan['activity_level'],
            '每周训练次数': plan['frequency'],
            '6 个月总成本 (元)': plan['cost'],
            '痰湿积分下降率': plan['effect_rate'],
            '预期积分下降': row['痰湿质'] * plan['effect_rate'],
            '预期最终积分': row['痰湿质'] * (1 - plan['effect_rate'])
        })

results_df = pd.DataFrame(results)
print(f"\n优化完成，共 {len(results_df)} 位患者")
print(results_df.head(10))

samples_123 = results_df[results_df['样本ID'].isin([1, 2, 3])].copy()
print("\n=== 样本 ID 1、2、3 最优干预方案 ===")
print(samples_123)

phlegm_groups = pd.cut(results_df['痰湿质积分'], bins=[0, 58, 61, 100], labels=['≤58 分', '59-61 分', '≥62 分'])
group_stats = results_df.groupby(phlegm_groups).agg({
    '样本ID': 'count',
    '6 个月总成本 (元)': 'mean',
    '痰湿积分下降率': 'mean'
}).round(2)
print("\n=== 按痰湿积分分组统计 ===")
print(group_stats)

age_stats = results_df.groupby('年龄组').agg({
    '样本ID': 'count',
    '6 个月总成本 (元)': 'mean',
    '活动干预强度': lambda x: x.mode().values[0] if len(x.mode()) > 0 else x.mean()
}).round(2)
print("\n=== 按年龄组分组统计 ===")
print(age_stats)

activity_groups = pd.cut(results_df['活动量表总分'], bins=[0, 39, 59, 100], labels=['低 (<40 分)', '中 (40-59 分)', '高 (≥60 分)'])
activity_stats = results_df.groupby(activity_groups).agg({
    '样本ID': 'count',
    '6 个月总成本 (元)': 'mean',
    '活动干预强度': 'mean',
    '每周训练次数': 'mean'
}).round(2)
print("\n=== 按活动能力分组统计 ===")
print(activity_stats)

results_df.to_excel(os.path.join(output_dir, '痰湿体质患者最优干预方案.xlsx'), index=False)
samples_123.to_excel(os.path.join(output_dir, '样本ID_1_2_3_干预方案.xlsx'), index=False)

summary_df = pd.DataFrame({
    '统计指标': ['患者总数', '平均成本 (元)', '平均下降率 (%)', '最低成本 (元)', '最高成本 (元)', '最低下降率 (%)', '最高下降率 (%)'],
    '数值': [
        len(results_df),
        results_df['6 个月总成本 (元)'].mean().round(2),
        (results_df['痰湿积分下降率'].mean() * 100).round(1),
        results_df['6 个月总成本 (元)'].min(),
        results_df['6 个月总成本 (元)'].max(),
        (results_df['痰湿积分下降率'].min() * 100).round(1),
        (results_df['痰湿积分下降率'].max() * 100).round(1)
    ]
})
summary_df.to_excel(os.path.join(output_dir, '优化方案统计摘要.xlsx'), index=False)

pattern_df = activity_stats.reset_index()
pattern_df.columns = ['活动能力分组', '患者数', '平均成本 (元)', '平均活动强度', '平均训练次数']
pattern_df.to_excel(os.path.join(output_dir, '患者特征 - 方案匹配规律.xlsx'), index=False)

print("\n=== 文件导出完成 ===")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

axes[0, 0].scatter(results_df['痰湿质积分'], results_df['6 个月总成本 (元)'], c=results_df['活动干预强度'], cmap='viridis', alpha=0.6, s=50)
axes[0, 0].set_xlabel('痰湿质积分', fontsize=12)
axes[0, 0].set_ylabel('6 个月总成本 (元)', fontsize=12)
axes[0, 0].set_title('痰湿积分与成本关系', fontsize=12)
plt.colorbar(axes[0, 0].collections[0], ax=axes[0, 0], label='活动强度等级')
axes[0, 0].grid(True, alpha=0.3)

axes[0, 1].scatter(results_df['痰湿质积分'], results_df['痰湿积分下降率'] * 100, c=results_df['6 个月总成本 (元)'], cmap='plasma', alpha=0.6, s=50)
axes[0, 1].set_xlabel('痰湿质积分', fontsize=12)
axes[0, 1].set_ylabel('痰湿积分下降率 (%)', fontsize=12)
axes[0, 1].set_title('痰湿积分与效果关系', fontsize=12)
plt.colorbar(axes[0, 1].collections[0], ax=axes[0, 1], label='成本 (元)')
axes[0, 1].grid(True, alpha=0.3)

tcm_dist = results_df['中医调理等级'].value_counts().sort_index()
axes[1, 0].bar(['基础调理', '中度调理', '强化调理'], tcm_dist.values, color=['#4E79A7', '#F28E2B', '#E15759'][:len(tcm_dist)])
axes[1, 0].set_ylabel('患者数', fontsize=12)
axes[1, 0].set_title('中医调理等级分布', fontsize=12)
axes[1, 0].grid(True, alpha=0.3, axis='y')

act_dist = results_df['活动干预强度'].value_counts().sort_index()
axes[1, 1].bar(['1 级强度', '2 级强度', '3 级强度'], act_dist.values, color=['#59A14F', '#76B7B2', '#4E79A7'][:len(act_dist)])
axes[1, 1].set_ylabel('患者数', fontsize=12)
axes[1, 1].set_title('活动干预强度分布', fontsize=12)
axes[1, 1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(os.path.join(output_dir, '问题三优化结果可视化.png'), dpi=150, bbox_inches='tight')
plt.close()

print("\n" + "="*80)
print("样本 ID 1、2、3 详细干预方案")
print("="*80)

for sample_id in [1, 2, 3]:
    sample_info = df[df['样本ID'] == sample_id].iloc[0]
    sample_plan = results_df[results_df['样本ID'] == sample_id].iloc[0]
    
    print(f"\n【样本 ID {sample_id}】")
    print(f"基本信息：")
    print(f"  - 痰湿质积分：{sample_info['痰湿质']}分")
    print(f"  - 年龄组：{sample_info['年龄组']}")
    print(f"  - 活动量表总分：{sample_info['活动量表总分（ADL总分+IADL总分）']}分")
    print(f"  - 高血脂症：{'是' if sample_info['高血脂症二分类标签']==1 else '否'}")
    print(f"\n最优干预方案：")
    print(f"  - 中医调理：{tcm_names[sample_plan['中医调理等级']]}")
    print(f"  - 活动干预：{act_names[sample_plan['活动干预强度']]}，{act_times[sample_plan['活动干预强度']]}/次")
    print(f"  - 训练频率：{sample_plan['每周训练次数']}次/周")
    print(f"\n预期效果：")
    print(f"  - 6 个月总成本：{sample_plan['6 个月总成本 (元)']}元")
    print(f"  - 痰湿积分下降率：{sample_plan['痰湿积分下降率']*100:.2f}%")
    print(f"  - 预期积分下降：{sample_plan['预期积分下降']:.2f}分")
    print(f"  - 预期最终积分：{sample_plan['预期最终积分']:.2f}分")

print("\n" + "="*80)
