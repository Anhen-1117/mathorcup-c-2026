import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, roc_curve, auc, confusion_matrix
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

plt.rcParams.update({
    'font.sans-serif': ['SimHei', 'DejaVu Sans'],
    'axes.unicode_minus': False
})

df = pd.read_excel('../1776658325832-问题一_清洗后数据.xlsx')

constitution_names = {
    1: '平和质', 2: '气虚质', 3: '阳虚质', 4: '阴虚质', 5: '痰湿质',
    6: '湿热质', 7: '血瘀质', 8: '气郁质', 9: '特禀质'
}

constitution_features = ['平和质', '气虚质', '阳虚质', '阴虚质', '痰湿质',
                         '湿热质', '血瘀质', '气郁质', '特禀质']

print("=" * 70)
print("九种体质对高血脂症发病风险贡献度分析")
print("=" * 70)

print("\n【1. 各体质类型的高血脂症患病率】")
constitution_stats = df.groupby('体质标签').agg({
    '高血脂症二分类标签': ['mean', 'count', 'sum'],
    '痰湿质': 'mean'
}).round(4)
constitution_stats.columns = ['患病率', '样本数', '患病人数', '痰湿质均值']
constitution_stats['体质名称'] = constitution_stats.index.map(constitution_names)
constitution_stats = constitution_stats[['体质名称', '样本数', '患病人数', '患病率', '痰湿质均值']]
constitution_stats = constitution_stats.sort_values('患病率', ascending=False)
print(constitution_stats.to_string(index=False))

print("\n【2. 九种体质积分与高血脂症的相关性】")
def get_sig(p):
    return '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else ''))

constitution_corr = []
for feat in constitution_features:
    corr, p_value = stats.pearsonr(df[feat], df['高血脂症二分类标签'])
    constitution_corr.append({
        '体质': feat,
        '相关系数': round(corr, 4),
        'p 值': round(p_value, 6),
        '显著性': get_sig(p_value)
    })

df_constitution_corr = pd.DataFrame(constitution_corr).sort_values('相关系数', key=abs, ascending=False)
print(df_constitution_corr.to_string(index=False))

print("\n【3. Logistic 回归分析各体质对发病风险的贡献】")

X_constitution = df[constitution_features]
y = df['高血脂症二分类标签']

scaler = StandardScaler()
X_constitution_scaled = scaler.fit_transform(X_constitution)

log_reg_constitution = LogisticRegression(max_iter=1000, random_state=42)
log_reg_constitution.fit(X_constitution_scaled, y)

constitution_coef = pd.DataFrame({
    '体质': constitution_features,
    '系数': log_reg_constitution.coef_[0],
    'OR值': np.exp(log_reg_constitution.coef_[0])
}).sort_values('OR值', ascending=False)
print(constitution_coef.to_string(index=False))

print("=" * 70)
print("多因素 Logistic 回归分析")
print("=" * 70)

selected_features = ['TG（甘油三酯）', '血尿酸', 'TC（总胆固醇）', 'LDL-C（低密度脂蛋白）', 'HDL-C（高密度脂蛋白）']
base_features = ['年龄组', '性别', '吸烟史', '饮酒史']
all_features = selected_features + constitution_features + base_features
target = '高血脂症二分类标签'

X = df[all_features].copy()
y = df[target].copy()

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

log_reg = LogisticRegression(max_iter=1000, random_state=42)
log_reg.fit(X_train, y_train)

y_pred = log_reg.predict(X_test)
y_pred_proba = log_reg.predict_proba(X_test)[:, 1]

print("\n【模型评估】")
print(f"训练集准确率：{log_reg.score(X_train, y_train):.4f}")
print(f"测试集准确率：{log_reg.score(X_test, y_test):.4f}")

cv_scores = cross_val_score(log_reg, X_scaled, y, cv=5, scoring='roc_auc')
print(f"交叉验证 AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

print("\n分类报告：")
print(classification_report(y_test, y_pred, target_names=['未确诊', '确诊']))

coef_df = pd.DataFrame({
    '特征': all_features,
    '系数': log_reg.coef_[0],
    'OR值': np.exp(log_reg.coef_[0])
}).sort_values('OR值', ascending=False)
print("\n【特征重要性（按 OR 值排序）】")
print(coef_df.to_string(index=False))

fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.3f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.savefig('../logistic/Logistic回归ROC曲线.png', dpi=300, bbox_inches='tight')
plt.close()

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['未确诊', '确诊'],
            yticklabels=['未确诊', '确诊'])
plt.xlabel('预测结果')
plt.ylabel('实际结果')
plt.title('混淆矩阵')
plt.savefig('../logistic/Logistic回归混淆矩阵.png', dpi=300, bbox_inches='tight')
plt.close()

top_features = coef_df.head(15)
plt.figure(figsize=(10, 6))
sns.barplot(x='OR值', y='特征', data=top_features, palette='viridis')
plt.xlabel('OR 值')
plt.ylabel('特征')
plt.title('特征重要性（按 OR 值排序）')
plt.tight_layout()
plt.savefig('../logistic/Logistic回归特征重要性.png', dpi=300, bbox_inches='tight')
plt.close()

print("\n分析完成！结果已保存到文件。")
