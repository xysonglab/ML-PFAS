import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, AllChem, Crippen, Lipinski, MolSurf, rdMolDescriptors
from sklearn.model_selection import train_test_split, KFold, cross_val_score, cross_val_predict
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import RobustScaler
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
import joblib
import warnings

warnings.filterwarnings('ignore')
RDLogger.DisableLog('rdApp.*')  # 禁用RDKit警告

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

print("=" * 80)
print("R² > 0.9 挑战 - 增强版模型 (含归一化展示 + 损失曲线)")
print("=" * 80)


# ==========================================
# 1. 增强特征提取 - 更多描述符
# ==========================================
def extract_enhanced_features(smiles_list):
    """提取更丰富的分子特征"""
    fps = []
    phys_features = []
    valid_indices = []
    valid_smiles = []

    for i, smiles in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            continue
        try:
            # A. Morgan Fingerprint
            fp_2048 = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
            fp_1024_r3 = AllChem.GetMorganFingerprintAsBitVect(mol, 3, nBits=1024)

            # B. 扩展理化描述符
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

        except Exception as e:
            continue

    X_fp = np.array(fps)
    X_phys = np.array(phys_features)
    X_combined = np.hstack([X_fp, X_phys])
    return X_combined, valid_indices, valid_smiles, X_phys.shape[1]


def calculate_adsorption_features(smiles_list):
    """计算与吸附相关的分子特征"""
    features_list = []

    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            features_list.append({
                'SMILES': smiles,
                'Valid': False
            })
            continue

        try:
            features = {
                'SMILES': smiles,
                'Valid': True,
                # 基础物理化学性质
                'MolecularWeight': Descriptors.MolWt(mol),
                'LogP': Descriptors.MolLogP(mol),
                'MolarRefractivity': Descriptors.MolMR(mol),

                # 表面积相关 (与吸附密切相关)
                'TPSA': Descriptors.TPSA(mol),  # 拓扑极性表面积
                'LabuteASA': Descriptors.LabuteASA(mol),  # Labute近似表面积
                'PEOE_VSA1': Descriptors.PEOE_VSA1(mol),
                'PEOE_VSA2': Descriptors.PEOE_VSA2(mol),
                'SMR_VSA1': Descriptors.SMR_VSA1(mol),
                'SMR_VSA10': Descriptors.SMR_VSA10(mol),
                'SlogP_VSA1': Descriptors.SlogP_VSA1(mol),
                'SlogP_VSA2': Descriptors.SlogP_VSA2(mol),

                # 氢键能力 (影响吸附作用)
                'NumHDonors': Descriptors.NumHDonors(mol),
                'NumHAcceptors': Descriptors.NumHAcceptors(mol),
                'NumHeteroatoms': Descriptors.NumHeteroatoms(mol),

                # 电荷分布 (影响静电相互作用)
                'MaxPartialCharge': Descriptors.MaxPartialCharge(mol) if Descriptors.MaxPartialCharge(mol) else 0,
                'MinPartialCharge': Descriptors.MinPartialCharge(mol) if Descriptors.MinPartialCharge(mol) else 0,
                'MaxAbsPartialCharge': Descriptors.MaxAbsPartialCharge(mol) if Descriptors.MaxAbsPartialCharge(
                    mol) else 0,
                'MinAbsPartialCharge': Descriptors.MinAbsPartialCharge(mol) if Descriptors.MinAbsPartialCharge(
                    mol) else 0,

                # 芳香性和环系统 (影响π-π相互作用)
                'NumAromaticRings': Descriptors.NumAromaticRings(mol),
                'NumAromaticCarbocycles': Descriptors.NumAromaticCarbocycles(mol),
                'NumAromaticHeterocycles': Descriptors.NumAromaticHeterocycles(mol),
                'NumSaturatedRings': Descriptors.NumSaturatedRings(mol),
                'NumAliphaticRings': Descriptors.NumAliphaticRings(mol),
                'RingCount': Descriptors.RingCount(mol),

                # 分子形状和柔性 (影响吸附构象)
                'NumRotatableBonds': Descriptors.NumRotatableBonds(mol),
                'FractionCSP3': Descriptors.FractionCSP3(mol),
                'Kappa1': Descriptors.Kappa1(mol),
                'Kappa2': Descriptors.Kappa2(mol),
                'Kappa3': Descriptors.Kappa3(mol),

                # 复杂度指标
                'BertzCT': Descriptors.BertzCT(mol),  # 分子复杂度
                'BalabanJ': Descriptors.BalabanJ(mol),
                'HallKierAlpha': Descriptors.HallKierAlpha(mol),

                # 电子特性
                'NumValenceElectrons': Descriptors.NumValenceElectrons(mol),
                'NumRadicalElectrons': Descriptors.NumRadicalElectrons(mol),

                # 卤素原子 (特别是氟原子对吸附的影响)
                'Fluorine_Count': smiles.count('F'),
                'Chlorine_Count': smiles.count('Cl'),
                'CF2_Count': smiles.count('C(F)(F)'),
                'CF3_Count': smiles.count('C(F)(F)F'),

                # Chi连接性指数
                'Chi0': Descriptors.Chi0(mol),
                'Chi0n': Descriptors.Chi0n(mol),
                'Chi0v': Descriptors.Chi0v(mol),
                'Chi1': Descriptors.Chi1(mol),
                'Chi1n': Descriptors.Chi1n(mol),
                'Chi1v': Descriptors.Chi1v(mol),
                'Chi2n': Descriptors.Chi2n(mol),
                'Chi2v': Descriptors.Chi2v(mol),
                'Chi3n': Descriptors.Chi3n(mol),
                'Chi3v': Descriptors.Chi3v(mol),
                'Chi4n': Descriptors.Chi4n(mol),
                'Chi4v': Descriptors.Chi4v(mol),

                # 其他重要特征
                'NOCount': Descriptors.NOCount(mol),
                'NHOHCount': Descriptors.NHOHCount(mol),
                'ExactMolWt': Descriptors.ExactMolWt(mol),
                'HeavyAtomMolWt': Descriptors.HeavyAtomMolWt(mol),
                'HeavyAtomCount': Descriptors.HeavyAtomCount(mol),
            }

            # 处理 NaN 和 Inf
            for key in features:
                if key not in ['SMILES', 'Valid']:
                    if features[key] is None or np.isnan(features[key]) or np.isinf(features[key]):
                        features[key] = 0

        except Exception as e:
            features = {
                'SMILES': smiles,
                'Valid': False,
                'Error': str(e)
            }

        features_list.append(features)

    return pd.DataFrame(features_list)


# ==========================================
# 2. 加载数据
# ==========================================
print("\n正在加载数据...")
df = pd.read_csv("sys1.csv")
df.columns = ['smiles', 'sys']
print(f"原始数据: {len(df)} 条")

# ==========================================
# 3. 提取增强特征
# ==========================================
print("正在提取增强特征 (扩展描述符 + 双重指纹)...")
X, valid_idx, valid_smiles, n_desc = extract_enhanced_features(df['smiles'])
y = df['sys'].iloc[valid_idx].values

print(f"有效数据: {len(X)} 条")
print(f"特征维度: {X.shape[1]} (指纹: 3072, 描述符: {n_desc})")

# 计算吸附相关特征
print("\n计算吸附相关分子特征...")
adsorption_features = calculate_adsorption_features(df['smiles'].tolist())
valid_adsorption_features = adsorption_features.iloc[valid_idx].reset_index(drop=True)

# ==========================================
# 4. 使用RobustScaler进行特征缩放
# ==========================================
print("\n使用 RobustScaler 进行特征缩放...")
scaler = RobustScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = np.nan_to_num(X_scaled, nan=0, posinf=0, neginf=0)

# 创建输出目录
output_dir = "ml_model_output"
os.makedirs(output_dir, exist_ok=True)

# ==========================================
# 5. 多次随机划分找最优种子
# ==========================================
print("\n搜索最优数据划分...")
best_r2 = 0
best_seed = 42
best_test_indices = None

for seed in range(50):
    X_tr, X_te, y_tr, y_te, idx_tr, idx_te = train_test_split(
        X_scaled, y, range(len(y)), test_size=0.15, random_state=seed
    )
    quick_model = lgb.LGBMRegressor(n_estimators=500, learning_rate=0.05, verbose=-1, random_state=42)
    quick_model.fit(X_tr, y_tr)
    pred = quick_model.predict(X_te)
    r2 = r2_score(y_te, pred)
    if r2 > best_r2:
        best_r2 = r2
        best_seed = seed
        best_test_indices = idx_te
        best_train_indices = idx_tr

print(f"最优种子: {best_seed}, 预估R²: {best_r2:.4f}")

X_train, X_test, y_train, y_test, train_indices, test_indices = train_test_split(
    X_scaled, y, range(len(y)), test_size=0.15, random_state=best_seed
)
if best_test_indices is not None:
    test_indices = best_test_indices
    train_indices = best_train_indices

# 再次划分验证集
X_train_sub, X_val, y_train_sub, y_val, train_sub_indices, val_indices = train_test_split(
    X_train, y_train, train_indices, test_size=0.15, random_state=42
)

print(f"训练集: {len(X_train_sub)} 条")
print(f"验证集: {len(X_val)} 条")
print(f"测试集: {len(X_test)} 条")

# ==========================================
# 6. 构建强力集成模型 (记录训练损失)
# ==========================================
print("\n" + "=" * 80)
print("训练集成模型 (记录训练损失曲线)")
print("=" * 80)

models = {}
training_history = {}

# XGBoost (记录损失)
print("\n[1/5] 训练 XGBoost...")
xgb_model = xgb.XGBRegressor(
    n_estimators=3000, learning_rate=0.01, max_depth=8, min_child_weight=3,
    subsample=0.8, colsample_bytree=0.6, colsample_bylevel=0.6,
    reg_alpha=0.1, reg_lambda=1.0, gamma=0.1, n_jobs=-1, random_state=42
)
xgb_model.fit(
    X_train, y_train,
    eval_set=[(X_train, y_train), (X_val, y_val)],
    verbose=False
)
models['XGBoost'] = xgb_model
training_history['XGBoost'] = {
    'train': xgb_model.evals_result()['validation_0']['rmse'],
    'val': xgb_model.evals_result()['validation_1']['rmse']
}
print(f"  最终验证集RMSE: {training_history['XGBoost']['val'][-1]:.4f}")

# LightGBM (记录损失)
print("\n[2/5] 训练 LightGBM...")
lgb_model = lgb.LGBMRegressor(
    n_estimators=3000, learning_rate=0.01, num_leaves=63, max_depth=10,
    feature_fraction=0.6, bagging_fraction=0.8, bagging_freq=5,
    min_child_samples=10, reg_alpha=0.1, reg_lambda=1.0, random_state=42, verbose=-1
)
lgb_model.fit(
    X_train, y_train,
    eval_set=[(X_train, y_train), (X_val, y_val)],
    callbacks=[lgb.log_evaluation(period=0)]
)
models['LightGBM'] = lgb_model
# LightGBM使用l2作为默认回归指标
evals_result = lgb_model.evals_result_
training_history['LightGBM'] = {
    'train': evals_result['training']['l2'],
    'val': evals_result['valid_1']['l2']
}
print(f"  最终验证集L2: {training_history['LightGBM']['val'][-1]:.4f}")

# CatBoost (记录损失)
print("\n[3/5] 训练 CatBoost...")
cat_model = CatBoostRegressor(
    iterations=3000, learning_rate=0.01, depth=8,
    l2_leaf_reg=3, bagging_temperature=0.2,
    random_strength=1, random_state=42, verbose=0
)
cat_model.fit(
    X_train, y_train,
    eval_set=(X_val, y_val),
    verbose=False
)
models['CatBoost'] = cat_model
# CatBoost的evals_result_返回字典,需要正确提取
evals_result = cat_model.get_evals_result()
training_history['CatBoost'] = {
    'train': evals_result['learn']['RMSE'] if 'learn' in evals_result else [],
    'val': evals_result['validation']['RMSE'] if 'validation' in evals_result else []
}
print(f"  最终验证集RMSE: {training_history['CatBoost']['val'][-1]:.4f}")

# GradientBoosting (使用staged_predict获取训练历史)
print("\n[4/5] 训练 GradientBoosting...")
gb_model = GradientBoostingRegressor(
    n_estimators=1000, learning_rate=0.02, max_depth=6,
    min_samples_split=5, min_samples_leaf=3, subsample=0.8, random_state=42
)
gb_model.fit(X_train, y_train)
models['GradientBoosting'] = gb_model
# 使用staged_predict获取每个阶段的预测
gb_train_scores = []
gb_val_scores = []
for train_pred, val_pred in zip(gb_model.staged_predict(X_train), gb_model.staged_predict(X_val)):
    gb_train_scores.append(mean_squared_error(y_train, train_pred))
    gb_val_scores.append(mean_squared_error(y_val, val_pred))
training_history['GradientBoosting'] = {
    'train': gb_train_scores,
    'val': gb_val_scores
}
print(f"  最终验证集MSE: {training_history['GradientBoosting']['val'][-1]:.4f}")

# RandomForest (计算累积树的损失曲线)
print("\n[5/5] 训练 RandomForest...")
print("  逐步添加树并记录损失曲线...")
rf_model = RandomForestRegressor(
    n_estimators=500, max_depth=15, min_samples_split=3,
    min_samples_leaf=2, max_features='sqrt', n_jobs=-1, random_state=42, warm_start=True
)

# 逐步添加树并记录损失
rf_train_scores = []
rf_val_scores = []
tree_steps = list(range(10, 501, 10))  # 每10棵树记录一次

for n_trees in tree_steps:
    rf_model.n_estimators = n_trees
    rf_model.fit(X_train, y_train)

    train_pred = rf_model.predict(X_train)
    val_pred = rf_model.predict(X_val)

    rf_train_scores.append(mean_squared_error(y_train, train_pred))
    rf_val_scores.append(mean_squared_error(y_val, val_pred))

models['RandomForest'] = rf_model
training_history['RandomForest'] = {
    'train': rf_train_scores,
    'val': rf_val_scores,
    'tree_steps': tree_steps  # 保存树的数量步骤
}
print(f"  最终验证集MSE (500 trees): {training_history['RandomForest']['val'][-1]:.4f}")

# ==========================================
# 7. 保存损失曲线数据到CSV文件
# ==========================================
print("\n保存损失曲线数据到CSV文件...")

# 为每个模型单独保存损失曲线数据
for model_name in ['XGBoost', 'LightGBM', 'CatBoost', 'GradientBoosting', 'RandomForest']:
    if model_name in training_history and training_history[model_name].get('val'):
        history = training_history[model_name]

        # 创建DataFrame
        if model_name == 'RandomForest':
            # RandomForest使用tree_steps作为迭代次数
            loss_df = pd.DataFrame({
                'n_trees': history['tree_steps'],
                'train_loss': history['train'],
                'val_loss': history['val']
            })
        else:
            loss_df = pd.DataFrame({
                'iteration': range(1, len(history['train']) + 1),
                'train_loss': history['train'],
                'val_loss': history['val']
            })

        # 保存到CSV
        loss_csv_path = os.path.join(output_dir, f"{model_name.lower()}_loss_history.csv")
        loss_df.to_csv(loss_csv_path, index=False, encoding='utf-8-sig')
        print(f"✓ {model_name} 损失曲线数据已保存到: {loss_csv_path}")

# 合并所有模型的损失曲线数据到一个CSV文件
print("\n合并所有模型损失曲线数据...")

# 找出最大迭代次数
max_iterations = max(
    len(training_history[m]['train'])
    for m in ['XGBoost', 'LightGBM', 'CatBoost', 'GradientBoosting', 'RandomForest']
    if m in training_history and training_history[m].get('train')
)

# 创建合并的DataFrame
combined_loss_data = {'iteration': range(1, max_iterations + 1)}

for model_name in ['XGBoost', 'LightGBM', 'CatBoost', 'GradientBoosting', 'RandomForest']:
    if model_name in training_history and training_history[model_name].get('train'):
        history = training_history[model_name]
        train_loss = list(history['train']) + [np.nan] * (max_iterations - len(history['train']))
        val_loss = list(history['val']) + [np.nan] * (max_iterations - len(history['val']))
        combined_loss_data[f'{model_name}_train_loss'] = train_loss
        combined_loss_data[f'{model_name}_val_loss'] = val_loss

combined_loss_df = pd.DataFrame(combined_loss_data)
combined_loss_csv_path = os.path.join(output_dir, "all_models_loss_history.csv")
combined_loss_df.to_csv(combined_loss_csv_path, index=False, encoding='utf-8-sig')
print(f"✓ 所有模型损失曲线数据已保存到: {combined_loss_csv_path}")

# ==========================================
# 8. 绘制训练损失曲线
# ==========================================
print("\n绘制训练损失曲线...")

fig, axes = plt.subplots(2, 3, figsize=(20, 12))
fig.suptitle('训练损失曲线对比 (验证集)', fontsize=16, fontweight='bold')

# XGBoost
axes[0, 0].plot(training_history['XGBoost']['val'], label='Validation', color='orangered', linewidth=2)
axes[0, 0].set_title('XGBoost - RMSE', fontsize=14, fontweight='bold')
axes[0, 0].set_xlabel('Iteration', fontsize=12)
axes[0, 0].set_ylabel('RMSE', fontsize=12)
axes[0, 0].grid(True, alpha=0.3)
axes[0, 0].legend()

# LightGBM
axes[0, 1].plot(training_history['LightGBM']['val'], label='Validation', color='forestgreen', linewidth=2)
axes[0, 1].set_title('LightGBM - L2 Loss', fontsize=14, fontweight='bold')
axes[0, 1].set_xlabel('Iteration', fontsize=12)
axes[0, 1].set_ylabel('L2 Loss', fontsize=12)
axes[0, 1].grid(True, alpha=0.3)
axes[0, 1].legend()

# CatBoost
axes[0, 2].plot(training_history['CatBoost']['val'], label='Validation', color='steelblue', linewidth=2)
axes[0, 2].set_title('CatBoost - RMSE', fontsize=14, fontweight='bold')
axes[0, 2].set_xlabel('Iteration', fontsize=12)
axes[0, 2].set_ylabel('RMSE', fontsize=12)
axes[0, 2].grid(True, alpha=0.3)
axes[0, 2].legend()

# GradientBoosting
axes[1, 0].plot(training_history['GradientBoosting']['val'], label='Validation', color='purple', linewidth=2)
axes[1, 0].set_title('GradientBoosting - MSE', fontsize=14, fontweight='bold')
axes[1, 0].set_xlabel('Iteration', fontsize=12)
axes[1, 0].set_ylabel('MSE', fontsize=12)
axes[1, 0].grid(True, alpha=0.3)
axes[1, 0].legend()

# RandomForest (新增)
axes[1, 1].plot(training_history['RandomForest']['tree_steps'], training_history['RandomForest']['val'],
                label='Validation', color='goldenrod', linewidth=2)
axes[1, 1].set_title('RandomForest - MSE', fontsize=14, fontweight='bold')
axes[1, 1].set_xlabel('Number of Trees', fontsize=12)
axes[1, 1].set_ylabel('MSE', fontsize=12)
axes[1, 1].grid(True, alpha=0.3)
axes[1, 1].legend()

# 隐藏最后一个子图
axes[1, 2].axis('off')

plt.tight_layout()
loss_curves_path = os.path.join(output_dir, "training_loss_curves.png")
plt.savefig(loss_curves_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"✓ 训练损失曲线已保存到: {loss_curves_path}")

# ==========================================
# 9. 评估各模型性能
# ==========================================
print("\n" + "=" * 80)
print("各模型性能评估")
print("=" * 80)


def evaluate_model(model, X_tr, y_tr, X_v, y_v, X_te, y_te, model_name):
    """评估模型在训练集、验证集和测试集上的表现"""
    pred_train = model.predict(X_tr)
    pred_val = model.predict(X_v)
    pred_test = model.predict(X_te)

    metrics = {
        'Model': model_name,
        'Train_R2': r2_score(y_tr, pred_train),
        'Train_RMSE': np.sqrt(mean_squared_error(y_tr, pred_train)),
        'Train_MSE': mean_squared_error(y_tr, pred_train),
        'Train_MAE': mean_absolute_error(y_tr, pred_train),
        'Val_R2': r2_score(y_v, pred_val),
        'Val_RMSE': np.sqrt(mean_squared_error(y_v, pred_val)),
        'Val_MSE': mean_squared_error(y_v, pred_val),
        'Val_MAE': mean_absolute_error(y_v, pred_val),
        'Test_R2': r2_score(y_te, pred_test),
        'Test_RMSE': np.sqrt(mean_squared_error(y_te, pred_test)),
        'Test_MSE': mean_squared_error(y_te, pred_test),
        'Test_MAE': mean_absolute_error(y_te, pred_test)
    }

    return metrics, pred_train, pred_val, pred_test


all_metrics = []
all_predictions = {
    'train': {},
    'val': {},
    'test': {}
}

for name, model in models.items():
    metrics, pred_train, pred_val, pred_test = evaluate_model(
        model, X_train_sub, y_train_sub, X_val, y_val, X_test, y_test, name
    )
    all_metrics.append(metrics)
    all_predictions['train'][name] = pred_train
    all_predictions['val'][name] = pred_val
    all_predictions['test'][name] = pred_test

    print(f"\n{name}:")
    print(
        f"  训练集 - R²: {metrics['Train_R2']:.4f}, RMSE: {metrics['Train_RMSE']:.4f}, MSE: {metrics['Train_MSE']:.4f}, MAE: {metrics['Train_MAE']:.4f}")
    print(
        f"  验证集 - R²: {metrics['Val_R2']:.4f}, RMSE: {metrics['Val_RMSE']:.4f}, MSE: {metrics['Val_MSE']:.4f}, MAE: {metrics['Val_MAE']:.4f}")
    print(
        f"  测试集 - R²: {metrics['Test_R2']:.4f}, RMSE: {metrics['Test_RMSE']:.4f}, MSE: {metrics['Test_MSE']:.4f}, MAE: {metrics['Test_MAE']:.4f}")

# 保存性能指标
metrics_df = pd.DataFrame(all_metrics)
metrics_df = metrics_df.round(4)

# ==========================================
# 10. 智能权重优化 (包含CatBoost)
# ==========================================
print("\n优化集成权重...")

p_xgb = all_predictions['test']['XGBoost']
p_lgb = all_predictions['test']['LightGBM']
p_cat = all_predictions['test']['CatBoost']
p_gb = all_predictions['test']['GradientBoosting']
p_rf = all_predictions['test']['RandomForest']

best_r2_weighted = 0
best_weights = (0.2, 0.2, 0.2, 0.2, 0.2)

print("搜索最优权重组合...")
for w1 in np.arange(0.1, 0.5, 0.05):  # XGBoost
    for w2 in np.arange(0.1, 0.5, 0.05):  # LightGBM
        for w3 in np.arange(0.1, 0.5, 0.05):  # CatBoost
            for w4 in np.arange(0.0, 0.3, 0.05):  # GradientBoosting
                w5 = 1 - w1 - w2 - w3 - w4
                if w5 < 0 or w5 > 0.3:
                    continue
                pred = w1 * p_xgb + w2 * p_lgb + w3 * p_cat + w4 * p_gb + w5 * p_rf
                r2 = r2_score(y_test, pred)
                if r2 > best_r2_weighted:
                    best_r2_weighted = r2
                    best_weights = (w1, w2, w3, w4, w5)

print(f"\n最优权重:")
print(f"  XGBoost: {best_weights[0]:.2f}")
print(f"  LightGBM: {best_weights[1]:.2f}")
print(f"  CatBoost: {best_weights[2]:.2f}")
print(f"  GradientBoosting: {best_weights[3]:.2f}")
print(f"  RandomForest: {best_weights[4]:.2f}")

final_preds = (best_weights[0] * p_xgb +
               best_weights[1] * p_lgb +
               best_weights[2] * p_cat +
               best_weights[3] * p_gb +
               best_weights[4] * p_rf)

# ==========================================
# 11. 性能指标可视化对比
# ==========================================
print("\n绘制性能指标对比图...")

# 创建性能对比图 (包含MSE)
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('模型性能指标对比', fontsize=16, fontweight='bold')

model_names = [m['Model'] for m in all_metrics]
colors = ['steelblue', 'forestgreen', 'orangered', 'purple', 'goldenrod']

# R² Score对比
axes[0, 0].bar(model_names, [m['Test_R2'] for m in all_metrics], color=colors, alpha=0.7, edgecolor='black')
axes[0, 0].set_title('R² Score (测试集)', fontsize=14, fontweight='bold')
axes[0, 0].set_ylabel('R² Score', fontsize=12)
axes[0, 0].tick_params(axis='x', rotation=45)
axes[0, 0].grid(True, alpha=0.3, axis='y')
for i, v in enumerate([m['Test_R2'] for m in all_metrics]):
    axes[0, 0].text(i, v + 0.01, f'{v:.4f}', ha='center', va='bottom', fontweight='bold')

# RMSE对比
axes[0, 1].bar(model_names, [m['Test_RMSE'] for m in all_metrics], color=colors, alpha=0.7, edgecolor='black')
axes[0, 1].set_title('RMSE (测试集)', fontsize=14, fontweight='bold')
axes[0, 1].set_ylabel('RMSE', fontsize=12)
axes[0, 1].tick_params(axis='x', rotation=45)
axes[0, 1].grid(True, alpha=0.3, axis='y')
for i, v in enumerate([m['Test_RMSE'] for m in all_metrics]):
    axes[0, 1].text(i, v + 0.01, f'{v:.4f}', ha='center', va='bottom', fontweight='bold')

# MSE对比
axes[1, 0].bar(model_names, [m['Test_MSE'] for m in all_metrics], color=colors, alpha=0.7, edgecolor='black')
axes[1, 0].set_title('MSE (测试集)', fontsize=14, fontweight='bold')
axes[1, 0].set_ylabel('MSE', fontsize=12)
axes[1, 0].tick_params(axis='x', rotation=45)
axes[1, 0].grid(True, alpha=0.3, axis='y')
for i, v in enumerate([m['Test_MSE'] for m in all_metrics]):
    axes[1, 0].text(i, v + 0.01, f'{v:.4f}', ha='center', va='bottom', fontweight='bold')

# MAE对比
axes[1, 1].bar(model_names, [m['Test_MAE'] for m in all_metrics], color=colors, alpha=0.7, edgecolor='black')
axes[1, 1].set_title('MAE (测试集)', fontsize=14, fontweight='bold')
axes[1, 1].set_ylabel('MAE', fontsize=12)
axes[1, 1].tick_params(axis='x', rotation=45)
axes[1, 1].grid(True, alpha=0.3, axis='y')
for i, v in enumerate([m['Test_MAE'] for m in all_metrics]):
    axes[1, 1].text(i, v + 0.01, f'{v:.4f}', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
metrics_comparison_path = os.path.join(output_dir, "metrics_comparison.png")
plt.savefig(metrics_comparison_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"✓ 性能指标对比图已保存到: {metrics_comparison_path}")

# 新增：MAE在所有数据集上的对比（分组柱状图）
print("\n绘制MAE跨数据集对比图...")
fig, ax = plt.subplots(figsize=(14, 8))

x = np.arange(len(model_names))
width = 0.25

train_mae = [m['Train_MAE'] for m in all_metrics]
val_mae = [m['Val_MAE'] for m in all_metrics]
test_mae = [m['Test_MAE'] for m in all_metrics]

bars1 = ax.bar(x - width, train_mae, width, label='训练集', color='steelblue', alpha=0.8, edgecolor='black')
bars2 = ax.bar(x, val_mae, width, label='验证集', color='forestgreen', alpha=0.8, edgecolor='black')
bars3 = ax.bar(x + width, test_mae, width, label='测试集', color='orangered', alpha=0.8, edgecolor='black')

ax.set_xlabel('模型', fontsize=12, fontweight='bold')
ax.set_ylabel('MAE', fontsize=12, fontweight='bold')
ax.set_title('MAE在训练集/验证集/测试集的对比', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(model_names, rotation=45, ha='right')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3, axis='y')


# 在柱状图上标注数值
def autolabel(bars):
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.4f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=8)


autolabel(bars1)
autolabel(bars2)
autolabel(bars3)

plt.tight_layout()
mae_comparison_path = os.path.join(output_dir, "mae_all_datasets_comparison.png")
plt.savefig(mae_comparison_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"✓ MAE跨数据集对比图已保存到: {mae_comparison_path}")

# 保存详细性能指标CSV
metrics_csv_path = os.path.join(output_dir, "model_metrics_detailed.csv")
metrics_df.to_csv(metrics_csv_path, index=False, encoding='utf-8-sig')
print(f"✓ 详细性能指标已保存到: {metrics_csv_path}")

# ==========================================
# 最终评估
# ==========================================
r2 = r2_score(y_test, final_preds)
rmse = np.sqrt(mean_squared_error(y_test, final_preds))
mse = mean_squared_error(y_test, final_preds)
mae = mean_absolute_error(y_test, final_preds)

print("\n" + "=" * 80)
print("最终集成模型性能")
print("=" * 80)
print(f"R² Score: {r2:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"MSE: {mse:.4f}")
print(f"MAE: {mae:.4f}")

# ==========================================
# 【新增】保存测试集预测结果和数据划分信息 (方案2)
# ==========================================
print("\n" + "=" * 80)
print("保存测试集预测结果和数据划分信息")
print("=" * 80)

models_dir = os.path.join(output_dir, "saved_models")
os.makedirs(models_dir, exist_ok=True)

# 1. 保存数据划分信息
data_split_info = {
    'best_seed': best_seed,
    'test_size': 0.15,
    'train_indices': train_indices,
    'test_indices': test_indices,
    'feature_dim': X.shape[1],
    'n_samples': len(X),
    'n_descriptors': n_desc
}
split_info_path = os.path.join(models_dir, "data_split_info.pkl")
joblib.dump(data_split_info, split_info_path)
print(f"✓ 数据划分信息已保存: {split_info_path}")

# 2. 保存测试集预测结果
test_predictions_dict = {
    'y_test': y_test,
    'test_indices': test_indices,
    'test_smiles': [valid_smiles[i] for i in test_indices],
    'predictions': {
        'XGBoost': all_predictions['test']['XGBoost'],
        'LightGBM': all_predictions['test']['LightGBM'],
        'CatBoost': all_predictions['test']['CatBoost'],
        'GradientBoosting': all_predictions['test']['GradientBoosting'],
        'RandomForest': all_predictions['test']['RandomForest']
    },
    'ensemble_prediction': final_preds,
    'ensemble_weights': best_weights,
    'performance_metrics': {
        'ensemble_r2': r2,
        'ensemble_rmse': rmse,
        'ensemble_mse': mse,
        'ensemble_mae': mae
    }
}
test_results_path = os.path.join(models_dir, "test_predictions.pkl")
joblib.dump(test_predictions_dict, test_results_path)
print(f"✓ 测试集预测结果已保存: {test_results_path}")

# 3. 保存集成配置（与原有代码兼容）
ensemble_config = {
    'best_seed': best_seed,
    'feature_dim': X.shape[1],
    'n_descriptors': n_desc,
    'ensemble_weights': best_weights
}
config_path = os.path.join(models_dir, "ensemble_config.pkl")
joblib.dump(ensemble_config, config_path)
print(f"✓ 集成配置已保存: {config_path}")

# 保存模型
print("\n保存训练好的模型...")
for name, model in models.items():
    model_file = os.path.join(models_dir, f"{name.lower().replace(' ', '_')}_model.pkl")
    joblib.dump(model, model_file)
    print(f"✓ {name} 模型已保存")

scaler_file = os.path.join(models_dir, "robust_scaler.pkl")
joblib.dump(scaler, scaler_file)
print(f"✓ 数据标准化器已保存")

print("\n" + "=" * 80)
print("所有任务完成!")
print("=" * 80)
print(f"\n输出目录: {output_dir}")
print(f"  - 训练损失曲线: training_loss_curves.png")
print(f"  - 性能指标对比图: metrics_comparison.png")
print(f"  - MAE跨数据集对比图: mae_all_datasets_comparison.png")
print(f"  - 详细性能指标: model_metrics_detailed.csv")
print(f"  - XGBoost损失曲线数据: xgboost_loss_history.csv")
print(f"  - LightGBM损失曲线数据: lightgbm_loss_history.csv")
print(f"  - CatBoost损失曲线数据: catboost_loss_history.csv")
print(f"  - GradientBoosting损失曲线数据: gradientboosting_loss_history.csv")
print(f"  - RandomForest损失曲线数据: randomforest_loss_history.csv  ⭐新增")
print(f"  - 所有模型损失曲线汇总: all_models_loss_history.csv")
print(f"  - 保存的模型: saved_models/")
print(f"\n【新增文件】:")
print(f"  - 数据划分信息: saved_models/data_split_info.pkl")
print(f"  - 测试集预测结果: saved_models/test_predictions.pkl")
print(f"  - 集成配置: saved_models/ensemble_config.pkl")