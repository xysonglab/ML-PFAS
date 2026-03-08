"""
Data Size 对 R² 的影响分析
===========================
策略:
  - 先运行 model_save2.py，获取完整训练结果和保存的模型/数据
  - 100% 数据时：直接加载 model_save2 保存的模型，确保结果完全一致
  - 10%~90%：使用相同的数据划分、scaler、超参数，仅子采样训练集
  - 测试集始终固定不变
  - 不使用集成模型，5个模型独立评估
"""

import os
import sys
import subprocess
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, AllChem
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import RobustScaler
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import joblib
import warnings

warnings.filterwarnings('ignore')
RDLogger.DisableLog('rdApp.*')

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

print("=" * 80)
print("Data Size 对 R² 的影响分析")
print("数据比例: 10%, 20%, 30%, ..., 100%")
print("100% 时直接调用 model_save2 保存的模型，确保完全一致")
print("=" * 80)

# ==========================================
# 0. 先运行 model_save2.py（如果尚未运行）
# ==========================================
model_save2_dir = "ml_model_output"
models_dir = os.path.join(model_save2_dir, "saved_models")

if not os.path.exists(os.path.join(models_dir, "test_predictions.pkl")):
    print("\n[Step 0] model_save2 结果不存在，先运行 model_save2.py ...")
    result = subprocess.run([sys.executable, "model_save2.py"], capture_output=False)
    if result.returncode != 0:
        print("ERROR: model_save2.py 运行失败，请先确保 model_save2.py 和 sys1.csv 在当前目录")
        sys.exit(1)
    print("model_save2.py 运行完成!\n")
else:
    print("\n[Step 0] 检测到 model_save2 已有保存结果，直接加载。")

# ==========================================
# 1. 加载 model_save2 保存的所有内容
# ==========================================
print("\n[Step 1] 加载 model_save2 保存的结果...")

# 加载数据划分信息
split_info = joblib.load(os.path.join(models_dir, "data_split_info.pkl"))
best_seed = split_info['best_seed']
train_indices = split_info['train_indices']
test_indices = split_info['test_indices']
n_desc = split_info['n_descriptors']
print(f"  最优种子: {best_seed}")
print(f"  训练集大小: {len(train_indices)}")
print(f"  测试集大小: {len(test_indices)}")

# 加载测试集预测结果 (100% 时直接使用)
test_predictions = joblib.load(os.path.join(models_dir, "test_predictions.pkl"))
y_test_saved = test_predictions['y_test']
predictions_100 = test_predictions['predictions']  # dict: model_name -> pred_array
print(f"  100% 各模型 Test R² (从 model_save2 直接读取):")
for name, pred in predictions_100.items():
    r2_val = r2_score(y_test_saved, pred)
    print(f"    {name}: {r2_val:.4f}")

# 加载 scaler
scaler = joblib.load(os.path.join(models_dir, "robust_scaler.pkl"))
print("  ✓ RobustScaler 已加载")

# 加载5个已保存的模型 (用于100%的评估)
saved_models = {}
model_names = ['XGBoost', 'LightGBM', 'CatBoost', 'GradientBoosting', 'RandomForest']

for name in model_names:
    # 尝试多种可能的文件名
    possible_names = [
        f"{name.lower()}_model.pkl",
        f"{name.lower().replace(' ', '_')}_model.pkl",
    ]
    if name == 'CatBoost':
        possible_names.append("catboostregressor_model.pkl")
    if name == 'GradientBoosting':
        possible_names.append("gradientboostingregressor_model.pkl")
    if name == 'RandomForest':
        possible_names.append("randomforestregressor_model.pkl")

    loaded = False
    for fname in possible_names:
        fpath = os.path.join(models_dir, fname)
        if os.path.exists(fpath):
            saved_models[name] = joblib.load(fpath)
            print(f"  ✓ {name} 模型已加载: {fname}")
            loaded = True
            break
    if not loaded:
        all_files = [f for f in os.listdir(models_dir) if f.endswith('.pkl') and 'model' in f.lower()]
        print(f"  ⚠ {name} 模型文件未找到! 可用文件: {all_files}")

# ==========================================
# 2. 重新提取特征 (与 model_save2 完全一致)
# ==========================================
print("\n[Step 2] 重新提取特征...")


def extract_enhanced_features(smiles_list):
    """与 model_save2 完全一致的特征提取"""
    fps = []
    phys_features = []
    valid_indices = []
    valid_smiles = []

    for i, smiles in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            continue
        try:
            fp_2048 = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
            fp_1024_r3 = AllChem.GetMorganFingerprintAsBitVect(mol, 3, nBits=1024)

            desc_list = [
                Descriptors.MolWt(mol),
                Descriptors.HeavyAtomMolWt(mol),
                Descriptors.ExactMolWt(mol),
                Descriptors.MolLogP(mol),
                Descriptors.MolMR(mol),
                Descriptors.TPSA(mol),
                Descriptors.LabuteASA(mol),
                Descriptors.NumHDonors(mol),
                Descriptors.NumHAcceptors(mol),
                Descriptors.NumRotatableBonds(mol),
                Descriptors.NumHeteroatoms(mol),
                Descriptors.NumAromaticRings(mol),
                Descriptors.NumAliphaticRings(mol),
                Descriptors.RingCount(mol),
                Descriptors.FractionCSP3(mol),
                Descriptors.MaxAbsPartialCharge(mol) if Descriptors.MaxAbsPartialCharge(mol) else 0,
                Descriptors.MinAbsPartialCharge(mol) if Descriptors.MinAbsPartialCharge(mol) else 0,
                Descriptors.MaxPartialCharge(mol) if Descriptors.MaxPartialCharge(mol) else 0,
                Descriptors.MinPartialCharge(mol) if Descriptors.MinPartialCharge(mol) else 0,
                smiles.count('F'),
                smiles.count('C(F)(F)'),
                smiles.count('C(F)(F)F'),
                Descriptors.NOCount(mol),
                Descriptors.NHOHCount(mol),
                Descriptors.NumValenceElectrons(mol),
                Descriptors.NumRadicalElectrons(mol),
                Descriptors.BalabanJ(mol),
                Descriptors.BertzCT(mol),
                Descriptors.Chi0(mol),
                Descriptors.Chi0n(mol),
                Descriptors.Chi0v(mol),
                Descriptors.Chi1(mol),
                Descriptors.Chi1n(mol),
                Descriptors.Chi1v(mol),
                Descriptors.Kappa1(mol),
                Descriptors.Kappa2(mol),
                Descriptors.Kappa3(mol),
                Descriptors.HallKierAlpha(mol),
                Descriptors.Ipc(mol),
            ]
            desc_list = [0 if (x is None or np.isnan(x) or np.isinf(x)) else x for x in desc_list]

            fps.append(np.concatenate([np.array(fp_2048), np.array(fp_1024_r3)]))
            phys_features.append(desc_list)
            valid_indices.append(i)
            valid_smiles.append(smiles)

        except Exception:
            continue

    X_fp = np.array(fps)
    X_phys = np.array(phys_features)
    X_combined = np.hstack([X_fp, X_phys])
    return X_combined, valid_indices, valid_smiles, X_phys.shape[1]


df = pd.read_csv("sys1.csv")
df.columns = ['smiles', 'sys']
print(f"原始数据: {len(df)} 条")

X, valid_idx, valid_smiles, n_desc_check = extract_enhanced_features(df['smiles'])
y = df['sys'].iloc[valid_idx].values
print(f"有效数据: {len(X)} 条, 特征维度: {X.shape[1]}")

# 使用 model_save2 保存的 scaler 进行变换
X_scaled = scaler.transform(X)
X_scaled = np.nan_to_num(X_scaled, nan=0, posinf=0, neginf=0)

# ==========================================
# 3. 恢复 model_save2 完全相同的数据划分
# ==========================================
print("\n[Step 3] 恢复 model_save2 的数据划分...")

train_idx_list = list(train_indices)
test_idx_list = list(test_indices)

X_train_all = X_scaled[train_idx_list]   # model_save2 中的 X_train
y_train_all = y[train_idx_list]           # model_save2 中的 y_train
X_test = X_scaled[test_idx_list]
y_test = y[test_idx_list]

# 验证测试集 y 与 model_save2 保存的一致
assert np.allclose(y_test, y_test_saved), "ERROR: 测试集 y 不一致!"
print(f"  ✓ 测试集 y 验证通过 (与 model_save2 完全一致)")

# 再次划分验证集 (与 model_save2 完全一致: test_size=0.15, random_state=42)
X_train_sub, X_val, y_train_sub, y_val = train_test_split(
    X_train_all, y_train_all, test_size=0.15, random_state=42
)

print(f"  X_train_all (模型训练用): {len(X_train_all)} 条")
print(f"  X_train_sub (去除验证后): {len(X_train_sub)} 条")
print(f"  X_val (验证集): {len(X_val)} 条")
print(f"  X_test (测试集): {len(X_test)} 条")

# ==========================================
# 4. 验证100%时加载的模型与保存的预测一致
# ==========================================
print("\n[Step 4] 验证加载的模型预测与保存结果一致...")

for name in model_names:
    if name in saved_models and name in predictions_100:
        pred_from_model = saved_models[name].predict(X_test)
        pred_from_saved = predictions_100[name]
        match = np.allclose(pred_from_model, pred_from_saved, atol=1e-6)
        r2_model = r2_score(y_test, pred_from_model)
        r2_saved = r2_score(y_test, pred_from_saved)
        status = "✓ 一致" if match else f"⚠ 微小差异 (R² diff={abs(r2_model-r2_saved):.8f})"
        print(f"  {name}: R²={r2_saved:.4f} {status}")

# ==========================================
# 5. 不同 data size 下训练各模型并记录 R²
# ==========================================
print("\n" + "=" * 80)
print("开始不同数据规模的训练实验")
print("=" * 80)

data_fractions = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

results_r2 = {name: [] for name in model_names}
results_rmse = {name: [] for name in model_names}
results_mae = {name: [] for name in model_names}
results_train_r2 = {name: [] for name in model_names}
train_sizes = []

# 固定打乱顺序，保证子集嵌套（小子集 ⊂ 大子集）
np.random.seed(42)
shuffled_indices = np.random.permutation(len(X_train_all))

for frac in data_fractions:
    n_samples = int(len(X_train_all) * frac)
    n_samples = max(n_samples, 10)
    train_sizes.append(n_samples)

    print(f"\n{'─'*60}")
    print(f"数据比例: {frac*100:.0f}% ({n_samples}/{len(X_train_all)} 条训练数据)")
    print(f"{'─'*60}")

    if frac >= 1.0:
        # ===== 100%: 直接使用 model_save2 保存的模型和预测 =====
        print("  → 100%: 直接使用 model_save2 保存的模型和预测结果")
        for name in model_names:
            pred_test = predictions_100[name]
            test_r2 = r2_score(y_test, pred_test)
            test_rmse = np.sqrt(mean_squared_error(y_test, pred_test))
            test_mae = mean_absolute_error(y_test, pred_test)

            if name in saved_models:
                pred_train = saved_models[name].predict(X_train_all)
                train_r2 = r2_score(y_train_all, pred_train)
            else:
                train_r2 = np.nan

            results_r2[name].append(test_r2)
            results_rmse[name].append(test_rmse)
            results_mae[name].append(test_mae)
            results_train_r2[name].append(train_r2)

            print(f"  {name}: Test R² = {test_r2:.4f}, RMSE = {test_rmse:.4f}, MAE = {test_mae:.4f} [model_save2]")
    else:
        # ===== 10%~90%: 子采样训练，超参数完全一致 =====
        subset_idx = shuffled_indices[:n_samples]
        X_train_subset = X_train_all[subset_idx]
        y_train_subset = y_train_all[subset_idx]

        for name in model_names:
            print(f"  训练 {name}...", end=" ", flush=True)

            if name == 'XGBoost':
                model = xgb.XGBRegressor(
                    n_estimators=3000, learning_rate=0.01, max_depth=8, min_child_weight=3,
                    subsample=0.8, colsample_bytree=0.6, colsample_bylevel=0.6,
                    reg_alpha=0.1, reg_lambda=1.0, gamma=0.1, n_jobs=-1, random_state=42
                )
                model.fit(
                    X_train_subset, y_train_subset,
                    eval_set=[(X_train_subset, y_train_subset), (X_val, y_val)],
                    verbose=False
                )
            elif name == 'LightGBM':
                model = lgb.LGBMRegressor(
                    n_estimators=3000, learning_rate=0.01, num_leaves=63, max_depth=10,
                    feature_fraction=0.6, bagging_fraction=0.8, bagging_freq=5,
                    min_child_samples=10, reg_alpha=0.1, reg_lambda=1.0, random_state=42, verbose=-1
                )
                model.fit(
                    X_train_subset, y_train_subset,
                    eval_set=[(X_train_subset, y_train_subset), (X_val, y_val)],
                    callbacks=[lgb.log_evaluation(period=0)]
                )
            elif name == 'CatBoost':
                model = CatBoostRegressor(
                    iterations=3000, learning_rate=0.01, depth=8,
                    l2_leaf_reg=3, bagging_temperature=0.2,
                    random_strength=1, random_state=42, verbose=0
                )
                model.fit(
                    X_train_subset, y_train_subset,
                    eval_set=(X_val, y_val),
                    verbose=False
                )
            elif name == 'GradientBoosting':
                model = GradientBoostingRegressor(
                    n_estimators=1000, learning_rate=0.02, max_depth=6,
                    min_samples_split=5, min_samples_leaf=3, subsample=0.8, random_state=42
                )
                model.fit(X_train_subset, y_train_subset)
            elif name == 'RandomForest':
                model = RandomForestRegressor(
                    n_estimators=500, max_depth=15, min_samples_split=3,
                    min_samples_leaf=2, max_features='sqrt', n_jobs=-1, random_state=42
                )
                model.fit(X_train_subset, y_train_subset)

            pred_test = model.predict(X_test)
            pred_train = model.predict(X_train_subset)

            test_r2 = r2_score(y_test, pred_test)
            test_rmse = np.sqrt(mean_squared_error(y_test, pred_test))
            test_mae = mean_absolute_error(y_test, pred_test)
            train_r2 = r2_score(y_train_subset, pred_train)

            results_r2[name].append(test_r2)
            results_rmse[name].append(test_rmse)
            results_mae[name].append(test_mae)
            results_train_r2[name].append(train_r2)

            print(f"Test R² = {test_r2:.4f}, RMSE = {test_rmse:.4f}, MAE = {test_mae:.4f}")

# ==========================================
# 6. 保存结果到CSV
# ==========================================
output_dir = "data_size_analysis_output"
os.makedirs(output_dir, exist_ok=True)

print("\n" + "=" * 80)
print("保存结果...")
print("=" * 80)

result_data = {
    'Data_Fraction(%)': [int(f * 100) for f in data_fractions],
    'Train_Samples': train_sizes,
}
for name in model_names:
    result_data[f'{name}_Test_R2'] = results_r2[name]
    result_data[f'{name}_Test_RMSE'] = results_rmse[name]
    result_data[f'{name}_Test_MAE'] = results_mae[name]
    result_data[f'{name}_Train_R2'] = results_train_r2[name]

result_df = pd.DataFrame(result_data)
result_df = result_df.round(4)

csv_path = os.path.join(output_dir, "data_size_vs_r2_results.csv")
result_df.to_csv(csv_path, index=False, encoding='utf-8-sig')
print(f"✓ 结果数据已保存到: {csv_path}")

# 打印汇总表
print("\n" + "=" * 80)
print("Data Size vs R² 汇总表 (测试集)")
print("=" * 80)
print(f"{'比例':>6} | {'样本数':>6}", end="")
for name in model_names:
    print(f" | {name:>18}", end="")
print()
print("-" * 115)
for i, frac in enumerate(data_fractions):
    tag = " ★" if frac == 1.0 else ""
    print(f"{int(frac*100):>5}% | {train_sizes[i]:>6}", end="")
    for name in model_names:
        print(f" | {results_r2[name][i]:>18.4f}", end="")
    print(tag)
print("\n★ = 直接来自 model_save2 保存的模型")

# ==========================================
# 7. 绘制图表
# ==========================================
print("\n绘制图表...")

colors = {
    'XGBoost': '#2196F3',
    'LightGBM': '#4CAF50',
    'CatBoost': '#FF5722',
    'GradientBoosting': '#9C27B0',
    'RandomForest': '#FF9800'
}
markers = {
    'XGBoost': 'o',
    'LightGBM': 's',
    'CatBoost': '^',
    'GradientBoosting': 'D',
    'RandomForest': 'v'
}
x_labels = [f"{int(f*100)}%" for f in data_fractions]

# ---- 图1: R² vs Data Size (主图) ----
fig, ax = plt.subplots(figsize=(12, 7))

for name in model_names:
    ax.plot(data_fractions, results_r2[name],
            marker=markers[name], color=colors[name],
            linewidth=2.5, markersize=8, label=name, alpha=0.9)

ax.set_xlabel('Training Data Fraction', fontsize=14, fontweight='bold')
ax.set_ylabel('Test R² Score', fontsize=14, fontweight='bold')
ax.set_title('Effect of Training Data Size on Model Performance (Test R²)\n'
             '100% = model_save2 saved models',
             fontsize=15, fontweight='bold')
ax.set_xticks(data_fractions)
ax.set_xticklabels(x_labels, fontsize=11)
ax.legend(fontsize=12, loc='lower right', framealpha=0.9)
ax.grid(True, alpha=0.3, linestyle='--')
ax.set_xlim(0.05, 1.05)

for name in model_names:
    final_r2 = results_r2[name][-1]
    ax.annotate(f'{final_r2:.4f}',
                xy=(1.0, final_r2),
                xytext=(10, 0), textcoords='offset points',
                fontsize=9, fontweight='bold', color=colors[name],
                ha='left', va='center')

ax.axvline(x=1.0, color='gray', linestyle=':', alpha=0.5)

plt.tight_layout()
fig_path = os.path.join(output_dir, "data_size_vs_r2.png")
plt.savefig(fig_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"✓ R² vs Data Size 主图: {fig_path}")

# ---- 图2: 多指标子图 (R², RMSE, MAE) ----
fig, axes = plt.subplots(1, 3, figsize=(20, 6))
fig.suptitle('Training Data Size Effect on Model Performance\n(100% = model_save2)',
             fontsize=16, fontweight='bold')

for name in model_names:
    axes[0].plot(data_fractions, results_r2[name],
                 marker=markers[name], color=colors[name],
                 linewidth=2, markersize=7, label=name, alpha=0.9)
axes[0].set_xlabel('Data Fraction', fontsize=12)
axes[0].set_ylabel('Test R²', fontsize=12)
axes[0].set_title('R² Score', fontsize=14, fontweight='bold')
axes[0].set_xticks(data_fractions)
axes[0].set_xticklabels(x_labels, fontsize=9, rotation=45)
axes[0].legend(fontsize=9)
axes[0].grid(True, alpha=0.3)

for name in model_names:
    axes[1].plot(data_fractions, results_rmse[name],
                 marker=markers[name], color=colors[name],
                 linewidth=2, markersize=7, label=name, alpha=0.9)
axes[1].set_xlabel('Data Fraction', fontsize=12)
axes[1].set_ylabel('Test RMSE', fontsize=12)
axes[1].set_title('RMSE', fontsize=14, fontweight='bold')
axes[1].set_xticks(data_fractions)
axes[1].set_xticklabels(x_labels, fontsize=9, rotation=45)
axes[1].legend(fontsize=9)
axes[1].grid(True, alpha=0.3)

for name in model_names:
    axes[2].plot(data_fractions, results_mae[name],
                 marker=markers[name], color=colors[name],
                 linewidth=2, markersize=7, label=name, alpha=0.9)
axes[2].set_xlabel('Data Fraction', fontsize=12)
axes[2].set_ylabel('Test MAE', fontsize=12)
axes[2].set_title('MAE', fontsize=14, fontweight='bold')
axes[2].set_xticks(data_fractions)
axes[2].set_xticklabels(x_labels, fontsize=9, rotation=45)
axes[2].legend(fontsize=9)
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
fig_path2 = os.path.join(output_dir, "data_size_vs_all_metrics.png")
plt.savefig(fig_path2, dpi=300, bbox_inches='tight')
plt.close()
print(f"✓ 多指标对比图: {fig_path2}")

# ---- 图3: 训练 vs 测试 R² (过拟合分析) ----
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('Train vs Test R² across Data Sizes (Overfitting Analysis)',
             fontsize=16, fontweight='bold')

for idx, name in enumerate(model_names):
    row, col = idx // 3, idx % 3
    ax = axes[row, col]

    ax.plot(data_fractions, results_train_r2[name],
            marker='o', color=colors[name], linewidth=2, markersize=7,
            label='Train R²', linestyle='--', alpha=0.7)
    ax.plot(data_fractions, results_r2[name],
            marker='s', color=colors[name], linewidth=2, markersize=7,
            label='Test R²', alpha=0.9)

    ax.fill_between(data_fractions,
                    results_train_r2[name], results_r2[name],
                    alpha=0.15, color=colors[name], label='Overfitting Gap')

    ax.set_xlabel('Data Fraction', fontsize=11)
    ax.set_ylabel('R²', fontsize=11)
    ax.set_title(f'{name}', fontsize=13, fontweight='bold')
    ax.set_xticks(data_fractions)
    ax.set_xticklabels(x_labels, fontsize=8, rotation=45)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

axes[1, 2].axis('off')

plt.tight_layout()
fig_path3 = os.path.join(output_dir, "data_size_overfitting_analysis.png")
plt.savefig(fig_path3, dpi=300, bbox_inches='tight')
plt.close()
print(f"✓ 过拟合分析图: {fig_path3}")

# ==========================================
# 8. 最终总结
# ==========================================
print("\n" + "=" * 80)
print("实验完成!")
print("=" * 80)
print("\n100% 数据量下各模型 Test R² (直接来自 model_save2, 完全一致):")
for name in model_names:
    print(f"  {name}: {results_r2[name][-1]:.4f}")

print(f"\n输出目录: {output_dir}/")
print(f"  - data_size_vs_r2_results.csv        (完整数据表)")
print(f"  - data_size_vs_r2.png                 (R² vs 数据比例主图)")
print(f"  - data_size_vs_all_metrics.png        (R²/RMSE/MAE 多指标对比)")
print(f"  - data_size_overfitting_analysis.png  (过拟合分析)")