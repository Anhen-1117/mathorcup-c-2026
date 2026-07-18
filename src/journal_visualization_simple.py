import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['SimHei', 'Arial'],
    'axes.unicode_minus': False,
    'figure.figsize': (8, 6),
    'figure.dpi': 300,
    'lines.linewidth': 1.5
})

df = pd.read_excel('../1776658325832-问题一_清洗后数据.xlsx')
print(f"数据形状: {df.shape}")

print("\n构建风险等级...")
hyperlipidemia = df.iloc[:, 31].values
tanwu_score = df.iloc[:, 6].values
activity_score = df.iloc[:, 23].values

risk_level = np.full(len(df), '低风险', dtype=object)
high_risk_mask = ((hyperlipidemia == 1) & (tanwu_score >= 40)) | ((tanwu_score >= 60) & (activity_score < 40))
risk_level[high_risk_mask] = '高风险'
mid_risk_mask = ((hyperlipidemia == 0) & (tanwu_score >= 40)) | ((hyperlipidemia == 1) & (tanwu_score < 40))
risk_level[mid_risk_mask] = '中风险'
df['风险等级'] = risk_level

risk_dist = df['风险等级'].value_counts()
print("\n风险等级分布：")
for level in ['低风险', '中风险', '高风险']:
    count = risk_dist.get(level, 0)
    pct = count / len(df) * 100
    print(f"   - {level}: {count} 例 ({pct:.1f}%)")

print("\n构建模型...")
feature_indices = [2, 3, 4, 5, 6, 7, 8, 9, 10, 16, 22, 23, 24, 25, 26, 27, 28, 29, 30, 33, 34, 35, 36]
feature_cols = [df.columns[i] for i in feature_indices]
X = df.iloc[:, feature_indices].values
y = df['风险等级'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)
test_acc = accuracy_score(y_test, y_pred)
print(f"\n测试集准确率：{test_acc:.4f}")

print("\n分类报告：")
report = classification_report(y_test, y_pred, output_dict=True)
print(classification_report(y_test, y_pred))

print("\n各风险等级性能指标：")
print("=" * 50)
for label in ['低风险', '中风险', '高风险']:
    if label in report:
        print(f"{label} - 精确率: {report[label]['precision']:.4f}, 召回率: {report[label]['recall']:.4f}")

cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='accuracy')
print(f"\n5折交叉验证准确率：{cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

feature_importance = pd.DataFrame({
    '特征': feature_cols,
    '重要性': model.feature_importances_
}).sort_values('重要性', ascending=False)

print("\n生成风险等级分布饼图...")
plt.figure(figsize=(8, 6))
risk_colors = ['#4CAF50', '#FFC107', '#F44336']
risk_dist = df['风险等级'].value_counts()
plt.pie(risk_dist.values, labels=risk_dist.index, autopct='%1.1f%%',
        startangle=90, colors=risk_colors, wedgeprops={'edgecolor': 'white', 'linewidth': 1})
plt.title('风险等级分布', fontsize=12, fontweight='bold')
plt.axis('equal')
plt.tight_layout()
plt.savefig('风险等级分布饼图_journal.png', dpi=300)
plt.close()

print("\n生成特征重要性条形图...")
plt.figure(figsize=(10, 6))
top_features = feature_importance.head(15)
plt.barh(range(len(top_features)), top_features['重要性'], color=sns.color_palette('viridis', len(top_features)))
plt.yticks(range(len(top_features)), top_features['特征'], fontsize=8)
plt.xlabel('特征重要性', fontsize=10)
plt.ylabel('特征', fontsize=10)
plt.title('特征重要性排序（Top 15）', fontsize=12, fontweight='bold')
plt.gca().invert_yaxis()
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig('特征重要性条形图_journal.png', dpi=300)
plt.close()

print("\n生成风险等级特征箱线图...")
key_features = ['痰湿质', 'TG（甘油三酯）', 'TC（总胆固醇）', 'LDL-C（低密度脂蛋白）', 'HDL-C（高密度脂蛋白）']
n_features = len(key_features)
fig, axes = plt.subplots(n_features, 1, figsize=(8, 2*n_features))

for i, feature in enumerate(key_features):
    sns.boxplot(x='风险等级', y=feature, data=df, order=['低风险', '中风险', '高风险'],
                ax=axes[i], palette=risk_colors)
    axes[i].set_title(feature, fontsize=10, fontweight='bold')
    axes[i].set_xlabel('风险等级', fontsize=8)
    axes[i].set_ylabel('数值', fontsize=8)

plt.tight_layout()
plt.savefig('风险等级特征箱线图_journal.png', dpi=300)
plt.close()

print("\n生成混淆矩阵...")
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test, y_pred, labels=['低风险', '中风险', '高风险'])
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['低风险', '中风险', '高风险'],
            yticklabels=['低风险', '中风险', '高风险'])
plt.xlabel('预测结果', fontsize=10)
plt.ylabel('实际结果', fontsize=10)
plt.title('混淆矩阵', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('混淆矩阵_journal.png', dpi=300)
plt.close()

print("\n生成痰湿体质风险分布...")
tanwu_patients = df[df.iloc[:, 1] == 5].copy()
tanwu_risk_dist = tanwu_patients['风险等级'].value_counts()

plt.figure(figsize=(8, 6))
plt.pie(tanwu_risk_dist.values, labels=tanwu_risk_dist.index, autopct='%1.1f%%',
        startangle=90, colors=risk_colors, wedgeprops={'edgecolor': 'white', 'linewidth': 1})
plt.title('痰湿体质患者风险分布', fontsize=12, fontweight='bold')
plt.axis('equal')
plt.tight_layout()
plt.savefig('痰湿体质风险分布_journal.png', dpi=300)
plt.close()

print("\n生成特征相关性热力图...")
key_feature_indices = [6, 26, 27, 25, 24, 29, 30, 23, 28]
key_feature_names = ['痰湿质', 'TG', 'TC', 'LDL-C', 'HDL-C', '血尿酸', 'BMI', '活动量表', '空腹血糖']

data_subset = df.iloc[:, key_feature_indices]
data_subset.columns = key_feature_names
corr_matrix = data_subset.corr()

plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0,
            fmt='.2f', linewidths=0.5, linecolor='white')
plt.title('关键特征相关性热力图', fontsize=12, fontweight='bold')
plt.xticks(rotation=45, ha='right', fontsize=8)
plt.yticks(fontsize=8)
plt.tight_layout()
plt.savefig('特征相关性热力图_journal.png', dpi=300)
plt.close()

print("\n生成风险等级与痰湿积分关系...")
plt.figure(figsize=(8, 6))
sns.violinplot(x='风险等级', y='痰湿质', data=df, order=['低风险', '中风险', '高风险'],
               palette=risk_colors, legend=True)
sns.boxplot(x='风险等级', y='痰湿质', data=df, order=['低风险', '中风险', '高风险'],
            color='white', width=0.2)
plt.xlabel('风险等级', fontsize=10)
plt.ylabel('痰湿质积分', fontsize=10)
plt.title('风险等级与痰湿质积分关系', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('风险等级与痰湿积分关系_journal.png', dpi=300)
plt.close()

print("\n生成统计数据...")
risk_stats = df.groupby('风险等级').agg({
    '痰湿质': ['mean', 'std'],
    'TG（甘油三酯）': ['mean', 'std'],
    'TC（总胆固醇）': ['mean', 'std'],
    'LDL-C（低密度脂蛋白）': ['mean', 'std'],
    'HDL-C（高密度脂蛋白）': ['mean', 'std'],
    '血尿酸': ['mean', 'std'],
    'BMI': ['mean', 'std']
}).round(2)

print("\n各风险等级统计数据：")
print(risk_stats)

risk_stats.to_excel('风险等级统计数据.xlsx')

print("\n特征重要性分析结果：")
print("=" * 50)
for i, (idx, row) in enumerate(feature_importance.iterrows()):
    print(f"{i+1}. {row['特征']}: {row['重要性']:.4f}")

top_5_features = feature_importance.head(5)['特征'].tolist()
print("\n核心特征组合识别：")
print("=" * 50)
print("前5个核心特征：")
for i, feature in enumerate(top_5_features):
    print(f"{i+1}. {feature}")

high_risk_df = df[df['风险等级'] == '高风险']
print("\n高风险样本中特征均值排序：")
feature_freq = {feature: high_risk_df[feature].mean() for feature in feature_cols}
sorted_freq = sorted(feature_freq.items(), key=lambda x: x[1], reverse=True)
for i, (feature, freq) in enumerate(sorted_freq[:10]):
    print(f"{i+1}. {feature}: {freq:.4f}")

print("\n阈值选取依据：")
print("=" * 50)
print("1. 高血脂症标签：使用二分类标签（0/1）作为基础风险指标")
print("2. 痰湿质积分阈值：")
print("   - 痰湿质≥60：中医诊断标准中的重度痰湿质")
print("   - 痰湿质≥40：中医诊断标准中的中度痰湿质")
print("3. 活动量表阈值：")
print("   - 活动量表<40：功能状态较差，增加风险")
print("4. 组合规则：")
print("   - 高风险：(高血脂症=1且痰湿质≥40) 或 (痰湿质≥60且活动量表<40)")
print("   - 中风险：(高血脂症=0且痰湿质≥40) 或 (高血脂症=1且痰湿质<40)")
print("   - 低风险：其他情况")

print("\n特征组合病理解释：")
print("=" * 50)
print("1. 痰湿质+脂质代谢指标（TG、TC、LDL-C）：")
print("   - 痰湿体质患者往往伴有脂质代谢紊乱")
print("   - 痰湿阻滞经络，影响气血运行，导致脂质堆积")
print("   - 脂质代谢异常是高血脂症的直接原因")
print("2. 痰湿质+活动量表：")
print("   - 活动量减少会加重痰湿堆积")
print("   - 缺乏运动导致气血不畅，痰湿难化")
print("   - 功能状态下降是痰湿体质加重的重要因素")
print("3. 痰湿质+血尿酸：")
print("   - 痰湿体质常伴有代谢综合征")
print("   - 尿酸代谢异常与痰湿阻滞密切相关")
print("   - 高尿酸是心血管疾病的独立危险因素")

print("\n生成求解流程图...")
plt.figure(figsize=(16, 12))

arrowprops = dict(arrowstyle="->", linewidth=1.5, color="#333333")

positions = {
    "数据加载": (0.5, 0.9),
    "风险等级构建": (0.5, 0.8),
    "特征选择": (0.3, 0.7),
    "数据标准化": (0.7, 0.7),
    "模型训练": (0.5, 0.6),
    "模型评估": (0.5, 0.5),
    "准确率": (0.3, 0.4),
    "精确率召回率": (0.5, 0.4),
    "F1分数": (0.7, 0.4),
    "交叉验证": (0.5, 0.3),
    "结果可视化": (0.5, 0.2)
}

for node, (x, y) in positions.items():
    if node in ["数据加载", "风险等级构建", "模型训练", "模型评估", "结果可视化"]:
        plt.text(x, y, node, ha='center', va='center',
                 bbox=dict(facecolor='#4CAF50', edgecolor='#333333', boxstyle='round,pad=0.5', alpha=0.8))
    elif node in ["特征选择", "数据标准化"]:
        plt.text(x, y, node, ha='center', va='center',
                 bbox=dict(facecolor='#2196F3', edgecolor='#333333', boxstyle='round,pad=0.5', alpha=0.8))
    else:
        plt.text(x, y, node, ha='center', va='center',
                 bbox=dict(facecolor='#FF9800', edgecolor='#333333', boxstyle='round,pad=0.5', alpha=0.8))

for (src, src_pos), (dst, dst_pos) in [
    ("数据加载", "风险等级构建"),
    ("风险等级构建", "特征选择"),
    ("风险等级构建", "数据标准化"),
    ("特征选择", "模型训练"),
    ("数据标准化", "模型训练"),
    ("模型训练", "模型评估"),
    ("模型评估", "准确率"),
    ("模型评估", "精确率召回率"),
    ("模型评估", "F1分数"),
    ("模型评估", "交叉验证"),
    ("交叉验证", "结果可视化")
]:
    plt.annotate("", xy=positions[dst], xytext=positions[src], arrowprops=arrowprops)

performance_text = (f"模型性能分析\n" +
                   f"- 测试集准确率: {test_acc:.4f}\n" +
                   f"- 交叉验证准确率: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})"
                   )
plt.text(0.5, 0.1, performance_text, ha='center', va='center',
         bbox=dict(facecolor='#E0E0E0', edgecolor='#333333', boxstyle='round,pad=1', alpha=0.8))

plt.title('风险等级预测求解流程图', fontsize=16, fontweight='bold')
plt.axis('off')
plt.tight_layout()
plt.savefig('求解流程图_journal.png', dpi=300)
plt.close()

print("\n分析完成！")
print("生成的可视化图像：")
print("1. 风险等级分布饼图_journal.png")
print("2. 特征重要性条形图_journal.png")
print("3. 风险等级特征箱线图_journal.png")
print("4. 混淆矩阵_journal.png")
print("5. 痰湿体质风险分布_journal.png")
print("6. 特征相关性热力图_journal.png")
print("7. 风险等级与痰湿积分关系_journal.png")
print("8. 求解流程图_journal.png")
