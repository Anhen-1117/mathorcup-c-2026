# MathorCup 2026 C题 — 中医体质分类与高血脂症风险预测

> **奖项：全国二等奖（国二）** · **赛题：C题** · **论文编号：CMC2608146**

基于 **Logistic 回归 → LASSO 特征筛选 → 随机森林 → 投票法集成** 的多模型对比框架，对中医体质问卷数据进行分析，构建高血脂症风险预测模型并制定个性化干预方案。

---

## 项目亮点

- **多模型对比评估**：Logistic 回归（基线）、LASSO 回归（特征筛选）、随机森林（特征重要性排序）、投票法集成（模型融合）
- **全链路能力**：从数据清洗→特征工程→模型训练→模型评估→干预方案设计的完整闭环
- **16 小时产出**：从读题到论文提交全流程，主笔 8,000+ 字竞赛论文
- **SCI/Nature 风格可视化**：全部图表使用 scienceplots 库，符合学术期刊规范

## 技术栈

| 模块 | 技术 |
|------|------|
| 数据分析 | Python + Pandas + NumPy |
| 建模 | scikit-learn（LogisticRegression / RandomForest / LASSO / Voting） |
| 可视化 | Matplotlib + Seaborn + SciencePlots |
| 统计 | Pearson 相关性分析、混淆矩阵、ROC 曲线 |
| 输出 | Excel（pandas.ExcelWriter）+ Nature 风格 PNG 图表 |

## 仓库结构

```
mathorcup-c-2026/
├── README.md
├── src/
│   ├── logistic_contribution_analysis.py   # Logistic 回归贡献分析
│   ├── merged_analysis.py                  # 集成分析（LASSO + 随机森林 + 投票法）
│   ├── journal_visualization_simple.py     # 期刊风格可视化图表
│   └── intervention_optimization.py        # 问题三：个性化干预方案优化
├── data/
│   └── 附件1：样例数据.xlsx                 # 原始问卷数据
├── paper/
│   ├── CMC2608146.pdf                      # 最终提交论文（PDF）
│   └── CMC2608146.docx                     # 最终提交论文（Word）
└── results/
    ├── Logistic回归ROC曲线.png
    ├── Logistic回归混淆矩阵.png
    ├── LASSO 回归分析_nature.png
    ├── LASSO 系数分布_nature.png
    ├── 随机森林特征重要性_nature.png
    ├── 随机森林特征重要性饼图_nature.png
    ├── Pearson 相关系数分析_nature.png
    ├── 投票法结果_nature.png
    ├── 特征相关性热力图_journal.png
    ├── 风险等级分布饼图_journal.png
    └── ...（共 18 张 SCI 风格图表）
```

## 问题与模型

### 问题一：高血脂症风险因素分析

- **数据清洗**：处理中医体质问卷数据，处理缺失值与异常值
- **特征工程**：构建痰湿质相关特征、高血脂症指标等多维度特征体系
- **相关性分析**：Pearson 相关系数热力图，识别与高血脂症显著相关的因素
- **Logistic 回归**：建立基线风险预测模型，输出 ROC 曲线（AUC）、混淆矩阵、特征重要性

### 问题二：多模型对比与集成

- **LASSO 回归**：L1 正则化特征筛选，系数路径图与收缩分析
- **随机森林**：特征重要性排序、累积重要性曲线、重要性饼图
- **投票法集成**：Soft Voting 融合 Logistic + 随机森林 + LASSO，提升泛化性能
- **模型评估**：ROC 曲线、混淆矩阵、精确率-召回率对比

### 问题三：个性化干预方案

- 基于模型预测结果，为高风险人群设计个性化干预方案
- 分析痰湿体质、饮食习惯、运动频率等可干预因素与风险等级的关系

## 快速运行

### 环境要求

```bash
pip install pandas numpy scikit-learn matplotlib seaborn
# 可选：SCI 风格图表
pip install SciencePlots
```

### 运行各模块

```bash
cd src

# 问题一：Logistic 回归分析
python logistic_contribution_analysis.py

# 问题一 + 二：集成分析（LASSO + 随机森林 + 投票法）
python merged_analysis.py

# 问题二：期刊风格可视化
python journal_visualization_simple.py

# 问题三：干预方案优化
python intervention_optimization.py
```

## 核心结果

| 模型 | AUC | 说明 |
|------|:---:|------|
| Logistic 回归 | 0.85+ | 基线模型，特征可解释性强 |
| LASSO 回归 | — | 筛选出关键特征，系数路径清晰 |
| 随机森林 | 0.90+ | 特征重要性排序，识别最强预测因子 |
| 投票法集成 | 0.90+ | Soft Voting 融合，综合性能最优 |

## 相关仓库

- [电工杯 B题 — 养老设施选址优化](https://github.com/Anhen-1117/dian-gong-bei-2026)
- [MathorCup D题 — 三维装箱优化](https://github.com/Anhen-1117/mathorcup-d-2026)
- [获奖证书](https://github.com/Anhen-1117/certificates)

## License

MIT
