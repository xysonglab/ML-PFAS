import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import warnings
import joblib
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, AllChem
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import lightgbm as lgb
from catboost import CatBoostRegressor
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.colors as mcolors
import matplotlib.ticker as ticker
from matplotlib.cm import ScalarMappable

warnings.filterwarnings('ignore')
RDLogger.DisableLog('rdApp.*')

print("=" * 80)
print("SHAP分析 - 从已保存的模型加载 (LightGBM & CatBoost)")
print("数据文件: sys1.csv")
print("=" * 80)

# ==========================================
# 配置参数
# ==========================================
OUTPUT_DIR = r"D:\ML\new413\new4\R\plot"  # 输出目录
MODELS_DIR = r"D:\ML\new413\new4\R\ml_model_output\saved_models"  # 模型目录
TOP_N_FEATURES = 15  # Top特征数量

# 创建输出目录
os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"\n✓ 输出目录: {OUTPUT_DIR}")
print(f"✓ 模型目录: {MODELS_DIR}")

# 设置图形参数 - 全局字体为Arial,全部加粗
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 600
plt.rcParams['savefig.dpi'] = 600

# 配色方案
COLOR_SCHEMES = {
    'viridis': plt.cm.viridis,
    'plasma': plt.cm.plasma,
    'coolwarm': plt.cm.coolwarm,
    'RdYlBu': plt.cm.RdYlBu,
    'RdBu_r': plt.cm.RdBu_r
}


# ==========================================
# 1. 特征提取函数 (与训练时完全一致)
# ==========================================
def extract_enhanced_features(smiles_list):
    """
    提取增强特征 - 与model_save.py中的函数完全一致
    包含: Morgan指纹(3072维) + 理化描述符(37维)
    """
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


# ==========================================
# 2. 加载数据
# ==========================================
print("\n" + "=" * 80)
print("步骤 1: 加载数据")
print("=" * 80)

# 读取数据
DATA_FILE = "sys1.csv"
possible_paths = [
    DATA_FILE,
    os.path.join(os.getcwd(), DATA_FILE),
    r"D:\ML\new413\new4\R\sys1.csv",
    "/mnt/user-data/uploads/sys1.csv"
]

df = None
for path in possible_paths:
    try:
        df = pd.read_csv(path)
        df.columns = ['smiles', 'sys']
        print(f"✓ 成功加载数据: {path}")
        break
    except:
        continue

if df is None:
    print("❌ 无法找到sys1.csv文件!")
    exit(1)

print(f"✓ 数据行数: {len(df)}")
print(f"✓ 目标值范围: [{df['sys'].min():.2f}, {df['sys'].max():.2f}]")

# ==========================================
# 3. 提取特征 (与训练时完全一致)
# ==========================================
print("\n正在提取增强特征 (与训练时一致)...")
X, valid_idx, valid_smiles, n_desc = extract_enhanced_features(df['smiles'])
y = df['sys'].iloc[valid_idx].values

print(f"✓ 有效样本数: {len(X)}")
print(f"✓ 特征维度: {X.shape[1]} (指纹: 3072, 描述符: {n_desc})")

# ==========================================
# 4. 加载模型和配置
# ==========================================
print("\n" + "=" * 80)
print("步骤 2: 加载已训练的模型")
print("=" * 80)

# 加载集成配置
config_path = os.path.join(MODELS_DIR, "ensemble_config.pkl")
try:
    ensemble_config = joblib.load(config_path)
    best_seed = ensemble_config['best_seed']
    print(f"✓ 加载集成配置: {config_path}")
    print(f"  - 最优种子: {best_seed}")
    print(f"  - 特征维度: {ensemble_config['feature_dim']}")
except Exception as e:
    print(f"⚠ 无法加载集成配置: {e}")
    print("  使用默认种子: 42")
    best_seed = 42

# 加载scaler
scaler_path = os.path.join(MODELS_DIR, "robust_scaler.pkl")
try:
    scaler = joblib.load(scaler_path)
    print(f"✓ 加载scaler: {scaler_path}")
except Exception as e:
    print(f"❌ 无法加载scaler: {e}")
    exit(1)

# 加载LightGBM模型
lgb_model_path = os.path.join(MODELS_DIR, "lightgbm_model.pkl")
try:
    lightgbm_model = joblib.load(lgb_model_path)
    print(f"✓ 加载LightGBM模型: {lgb_model_path}")
except Exception as e:
    print(f"❌ 无法加载LightGBM模型: {e}")
    exit(1)

# 加载CatBoost模型
cat_model_path = os.path.join(MODELS_DIR, "catboost_model.pkl")
try:
    catboost_model = joblib.load(cat_model_path)
    print(f"✓ 加载CatBoost模型: {cat_model_path}")
except Exception as e:
    print(f"❌ 无法加载CatBoost模型: {e}")
    exit(1)

xgb_model_path = os.path.join(MODELS_DIR, "xgboost_model.pkl")
try:
    xgboost_model = joblib.load(xgb_model_path)
    print(f"✓ 加载XGBoost模型: {xgb_model_path}")
except Exception as e:
    print(f"❌ 无法加载XGBoost模型: {e}")
    exit(1)

random_model_path = os.path.join(MODELS_DIR, "randomforest_model.pkl")
try:
    randomforest_model = joblib.load(random_model_path)
    print(f"✓ 加载randomforest模型: {random_model_path}")
except Exception as e:
    print(f"❌ 无法加载randomforest模型: {e}")
    exit(1)

gradient_model_path = os.path.join(MODELS_DIR, "gradientboosting_model.pkl")
try:
    gradient_model = joblib.load(gradient_model_path)
    print(f"✓ 加载gradientboosting模型: {gradient_model_path}")
except Exception as e:
    print(f"❌ 无法加载gradientboosting模型: {e}")
    exit(1)

# ==========================================
# 5. 数据预处理和划分 (与训练时完全一致)
# ==========================================
print("\n数据预处理...")
X_scaled = scaler.transform(X)
X_scaled = np.nan_to_num(X_scaled, nan=0, posinf=0, neginf=0)

print(f"使用训练时的数据划分 (seed={best_seed}, test_size=0.15)...")
X_train, X_test, y_train, y_test, train_indices, test_indices = train_test_split(
    X_scaled, y, range(len(y)), test_size=0.15, random_state=best_seed
)

# 获取对应的SMILES
train_smiles = [valid_smiles[i] for i in train_indices]
test_smiles = [valid_smiles[i] for i in test_indices]

print(f"✓ 训练集: {len(X_train)} 条")
print(f"✓ 测试集: {len(X_test)} 条")

# ==========================================
# 6. 模型预测和评估
# ==========================================
print("\n" + "=" * 80)
print("步骤 3: 模型评估")
print("=" * 80)

# LightGBM预测
lightgbm_pred_train = lightgbm_model.predict(X_train)
lightgbm_pred_test = lightgbm_model.predict(X_test)
lightgbm_r2_train = r2_score(y_train, lightgbm_pred_train)
lightgbm_r2_test = r2_score(y_test, lightgbm_pred_test)
lightgbm_rmse_test = np.sqrt(mean_squared_error(y_test, lightgbm_pred_test))
lightgbm_mae_test = mean_absolute_error(y_test, lightgbm_pred_test)

print(f"\nLightGBM:")
print(f"  训练集 R²: {lightgbm_r2_train:.4f}")
print(f"  测试集 R²: {lightgbm_r2_test:.4f}")
print(f"  测试集 RMSE: {lightgbm_rmse_test:.4f}")
print(f"  测试集 MAE: {lightgbm_mae_test:.4f}")

# CatBoost预测
catboost_pred_train = catboost_model.predict(X_train)
catboost_pred_test = catboost_model.predict(X_test)
catboost_r2_train = r2_score(y_train, catboost_pred_train)
catboost_r2_test = r2_score(y_test, catboost_pred_test)
catboost_rmse_test = np.sqrt(mean_squared_error(y_test, catboost_pred_test))
catboost_mae_test = mean_absolute_error(y_test, catboost_pred_test)

print(f"\nCatBoost:")
print(f"  训练集 R²: {catboost_r2_train:.4f}")
print(f"  测试集 R²: {catboost_r2_test:.4f}")
print(f"  测试集 RMSE: {catboost_rmse_test:.4f}")
print(f"  测试集 MAE: {catboost_mae_test:.4f}")

xgboost_pred_train = xgboost_model.predict(X_train)
xgboost_pred_test = xgboost_model.predict(X_test)
xgboost_r2_train = r2_score(y_train, xgboost_pred_train)
xgboost_r2_test = r2_score(y_test, xgboost_pred_test)
xgboost_rmse_test = np.sqrt(mean_squared_error(y_test, xgboost_pred_test))
xgboost_mae_test = mean_absolute_error(y_test, xgboost_pred_test)

print(f"\nxgboost:")
print(f"  训练集 R²: {xgboost_r2_train:.4f}")
print(f"  测试集 R²: {xgboost_r2_test:.4f}")
print(f"  测试集 RMSE: {xgboost_rmse_test:.4f}")
print(f"  测试集 MAE: {xgboost_mae_test:.4f}")

randomforest_pred_train = randomforest_model.predict(X_train)
randomforest_pred_test = randomforest_model.predict(X_test)
randomforest_r2_train = r2_score(y_train, randomforest_pred_train)
randomforest_r2_test = r2_score(y_test, randomforest_pred_test)
randomforest_rmse_test = np.sqrt(mean_squared_error(y_test, randomforest_pred_test))
randomforest_mae_test = mean_absolute_error(y_test, randomforest_pred_test)

print(f"\nrandomforest:")
print(f"  训练集 R²: {randomforest_r2_train:.4f}")
print(f"  测试集 R²: {randomforest_r2_test:.4f}")
print(f"  测试集 RMSE: {randomforest_rmse_test:.4f}")
print(f"  测试集 MAE: {randomforest_mae_test:.4f}")

gradient_pred_train = gradient_model.predict(X_train)
gradient_pred_test = gradient_model.predict(X_test)
gradient_r2_train = r2_score(y_train, gradient_pred_train)
gradient_r2_test = r2_score(y_test, gradient_pred_test)
gradient_rmse_test = np.sqrt(mean_squared_error(y_test, gradient_pred_test))
gradient_mae_test = mean_absolute_error(y_test, gradient_pred_test)

print(f"\ngradient:")
print(f"  训练集 R²: {gradient_r2_train:.4f}")
print(f"  测试集 R²: {gradient_r2_test:.4f}")
print(f"  测试集 RMSE: {gradient_rmse_test:.4f}")
print(f"  测试集 MAE: {gradient_mae_test:.4f}")
# # 保存性能指标
# model_performance = pd.DataFrame({
#     'Model': ['LightGBM', 'CatBoost'],
#     'Train_R2': [lightgbm_r2_train, catboost_r2_train],
#     'Test_R2': [lightgbm_r2_test, catboost_r2_test],
#     'Test_RMSE': [lightgbm_rmse_test, catboost_rmse_test],
#     'Test_MAE': [lightgbm_mae_test, catboost_mae_test]
# })
# model_perf_path = os.path.join(OUTPUT_DIR, "model_performance.csv")
# model_performance.to_csv(model_perf_path, index=False)
# print(f"\n✓ 模型性能已保存: {model_perf_path}")

# 保存测试集预测结果
# test_predictions = pd.DataFrame({
#     'SMILES': test_smiles,
#     'True_Value': y_test,
#     'LightGBM_Pred': lightgbm_pred_test,
#     'CatBoost_Pred': catboost_pred_test,
#     'LightGBM_Error': y_test - lightgbm_pred_test,
#     'CatBoost_Error': y_test - catboost_pred_test
# })
# test_pred_path = os.path.join(OUTPUT_DIR, "test_predictions.csv")
# test_predictions.to_csv(test_pred_path, index=False)
# print(f"✓ 测试集预测结果已保存: {test_pred_path}")

# ==========================================
# 7. SHAP分析 (仅针对理化描述符)
# ==========================================
print("\n" + "=" * 80)
print("步骤 4: SHAP分析 (理化描述符)")
print("=" * 80)

# 提取理化描述符部分 (最后37维)
n_fingerprint = 3072
X_train_desc = X_train[:, n_fingerprint:]
X_test_desc = X_test[:, n_fingerprint:]

# 创建描述符名称
descriptor_names = [
    'MolecularWeight', 'HeavyAtomMolWt', 'ExactMolWt', 'LogP', 'MolarRefractivity',
    'TPSA', 'LabuteASA', 'NumHDonors', 'NumHAcceptors', 'NumRotatableBonds',
    'NumHeteroatoms', 'NumAromaticRings', 'NumAliphaticRings', 'RingCount', 'FractionCSP3',
    'MaxAbsPartialCharge', 'MinAbsPartialCharge', 'MaxPartialCharge', 'MinPartialCharge',
    'Fluorine_Count', 'CF2_Count', 'CF3_Count', 'NOCount', 'NHOHCount',
    'NumValenceElectrons', 'NumRadicalElectrons', 'BalabanJ', 'BertzCT',
    'Chi0', 'Chi0n', 'Chi0v', 'Chi1', 'Chi1n', 'Chi1v',
    'Kappa1', 'Kappa2', 'Kappa3', 'HallKierAlpha', 'Ipc'
]

# 转换为DataFrame
X_test_desc_df = pd.DataFrame(X_test_desc, columns=descriptor_names[:X_test_desc.shape[1]])


def get_top_features_shap(model, X_data_full, X_data_df, feature_names, top_n=20, model_name="Model"):
    """使用SHAP获取最重要的特征"""
    print(f"\n计算 {model_name} 的SHAP值...")
    # 使用完整数据初始化explainer,禁用可加性检查
    explainer = shap.TreeExplainer(model, check_additivity=False)
    # 对完整特征计算SHAP值
    shap_values = explainer.shap_values(X_data_full)

    # 提取描述符部分的SHAP值
    shap_values_desc = shap_values[:, n_fingerprint:]

    # 计算平均绝对SHAP值
    mean_abs_shap = np.abs(shap_values_desc).mean(axis=0)

    # 创建特征重要性DataFrame
    feature_importance = pd.DataFrame({
        'Feature': feature_names,
        'Mean_Abs_SHAP': mean_abs_shap
    }).sort_values('Mean_Abs_SHAP', ascending=False)

    # 选取前N个特征
    top_features = feature_importance.head(top_n)
    top_feature_names = top_features['Feature'].tolist()

    print(f"✓ 已选取前 {top_n} 个最重要的描述符特征")
    print(f"  (注: 已禁用可加性检查以避免数值误差)")
    return shap_values_desc, feature_importance, top_feature_names


# 获取两个模型的Top特征
lightgbm_shap_values, lightgbm_importance, lightgbm_top_features = get_top_features_shap(
    lightgbm_model, X_test, X_test_desc_df, descriptor_names[:X_test_desc.shape[1]],
    TOP_N_FEATURES, "LightGBM"
)

catboost_shap_values, catboost_importance, catboost_top_features = get_top_features_shap(
    catboost_model, X_test, X_test_desc_df, descriptor_names[:X_test_desc.shape[1]],
    TOP_N_FEATURES, "CatBoost"
)

xgboost_shap_values, xgboost_importance, xgboost_top_features = get_top_features_shap(
    xgboost_model, X_test, X_test_desc_df, descriptor_names[:X_test_desc.shape[1]],
    TOP_N_FEATURES, "XGBoost"
)

randomforest_shap_values, randomforest_importance, randomforest_top_features = get_top_features_shap(
    randomforest_model, X_test, X_test_desc_df, descriptor_names[:X_test_desc.shape[1]],
    TOP_N_FEATURES, "randomforest"
)

gradient_shap_values, gradient_importance, gradient_top_features = get_top_features_shap(
    gradient_model, X_test, X_test_desc_df, descriptor_names[:X_test_desc.shape[1]],
    TOP_N_FEATURES, "gradientingBoost"
)

# 保存特征重要性
all_importance_path = os.path.join(OUTPUT_DIR, "feature_importance_all.csv")
lightgbm_importance['Model'] = 'LightGBM'
catboost_importance['Model'] = 'CatBoost'
xgboost_importance['Model'] = 'XGBoost'
randomforest_importance['Model'] = 'randomforest'
gradient_importance['Model'] = 'gradientingboost'
all_importance = pd.concat([lightgbm_importance, catboost_importance, xgboost_importance, gradient_importance, randomforest_importance])
all_importance.to_csv(all_importance_path, index=False)
print(f"\n✓ 特征重要性已保存: {all_importance_path}")

# 保存Top特征对比
top_comparison = pd.DataFrame({
    'Rank': range(1, TOP_N_FEATURES + 1),
    'LightGBM_Feature': lightgbm_top_features,
    'LightGBM_SHAP': [lightgbm_importance[lightgbm_importance['Feature'] == f]['Mean_Abs_SHAP'].values[0]
                      for f in lightgbm_top_features],
    'CatBoost_Feature': catboost_top_features,
    'CatBoost_SHAP': [catboost_importance[catboost_importance['Feature'] == f]['Mean_Abs_SHAP'].values[0]
                      for f in catboost_top_features],
    'XGBoost_Feature': xgboost_top_features,
    'XGBoost_SHAP': [xgboost_importance[xgboost_importance['Feature'] == f]['Mean_Abs_SHAP'].values[0]
                      for f in xgboost_top_features],
    'RandomForest_Feature': randomforest_top_features,
    'RandomForest_SHAP': [randomforest_importance[randomforest_importance['Feature'] == f]['Mean_Abs_SHAP'].values[0]
                     for f in randomforest_top_features],
    'gradientboost_Feature': gradient_top_features,
    'gradientboost_SHAP': [gradient_importance[gradient_importance['Feature'] == f]['Mean_Abs_SHAP'].values[0]
                          for f in gradient_top_features]
})
top_comparison_path = os.path.join(OUTPUT_DIR, f"top{TOP_N_FEATURES}_features.csv")
top_comparison.to_csv(top_comparison_path, index=False)
print(f"✓ Top{TOP_N_FEATURES}特征对比已保存: {top_comparison_path}")


# ==========================================
# 8. SHAP可视化函数
# ==========================================

def create_optimized_cmap(base_cmap, start=0.2, end=0.9):
    """创建优化的颜色映射"""
    colors = base_cmap(np.linspace(start, end, 256))
    return LinearSegmentedColormap.from_list('optimized', colors)


def plot_shap_barplot_with_rose(shap_values, X_data, feature_names, model_name, color_scheme='viridis'):
    """
    绘制条形图+玫瑰图组合 (正方形条形图)
    特征名称在最右边,与条形精确对齐,无边框,不遮挡图形
    """
    # 选择指定特征的数据
    feature_indices = [list(X_data.columns).index(f) for f in feature_names]
    shap_subset = shap_values[:, feature_indices]

    # 计算排序信息
    mean_abs_shap = np.abs(shap_subset).mean(axis=0)
    shap_series = pd.Series(mean_abs_shap, index=feature_names)
    shap_series.sort_values(ascending=False, inplace=True)
    sorted_features = shap_series.index.tolist()
    sorted_shap_values = shap_series.values

    # 玫瑰图数据
    base_length, fixed_increment, colored_ring_width = 2.0, 0.25, 1.0
    num_vars = len(feature_names)
    one_oclock_offset = np.pi / 21
    percentages = (sorted_shap_values / sorted_shap_values.sum()) * 100
    widths = (sorted_shap_values / sorted_shap_values.sum()) * 2 * np.pi
    thetas = np.cumsum([0] + widths[:-1].tolist()) - one_oclock_offset
    total_lengths = [base_length + i * fixed_increment for i in range(num_vars)]
    inner_heights = [max(0, tl - colored_ring_width) for tl in total_lengths]
    inner_colors = ['#F5F5F5', '#FFFFFF'] * (num_vars // 2 + 1)

    # 颜色映射
    cmap_base = COLOR_SCHEMES[color_scheme]
    cmap = create_optimized_cmap(cmap_base)
    color_norm = mcolors.Normalize(vmin=np.quantile(sorted_shap_values, 0.25),
                                   vmax=np.quantile(sorted_shap_values, 0.75))
    colors = cmap(color_norm(sorted_shap_values))

    # 正方形图形布局 - 为右侧特征名称预留更多空间
    fig_width = 10
    fig_height = 8
    fig = plt.figure(figsize=(fig_width, fig_height), dpi=600, facecolor='white')

    # 布局参数 - 主图占据左侧大部分空间
    left_margin = 0.10
    right_margin = 0.22  # 为特征名称预留右侧空间
    bottom_margin = 0.12
    top_margin = 0.12
    main_plot_width = 1 - left_margin - right_margin
    plot_bottom = bottom_margin
    plot_height = 1 - bottom_margin - top_margin

    # 左侧颜色条
    cbar_left = 0.02
    colorbar_width = 0.015
    ax_cbar = fig.add_axes([cbar_left, plot_bottom, colorbar_width, plot_height])
    sm = ScalarMappable(cmap=cmap, norm=color_norm)
    cbar = fig.colorbar(sm, cax=ax_cbar, orientation='vertical')
    cbar.set_ticks([])
    cbar.ax.yaxis.set_ticks_position('left')
    ax_cbar.text(-3.5, 0.98, 'High', transform=ax_cbar.transAxes, ha='center',
                 va='bottom', fontsize=24, fontweight='bold', family='Arial')
    ax_cbar.text(-3.5, 0.02, 'Low', transform=ax_cbar.transAxes, ha='center',
                 va='top', fontsize=24, fontweight='bold', family='Arial')
    cbar.outline.set_visible(False)
    ax_cbar.text(-3.5, 0.5, 'SHAP Value', transform=ax_cbar.transAxes,
                 fontsize=28, rotation=90, va='center', fontweight='bold', ha='center', family='Arial')
    ax_cbar.set_facecolor('white')

    # 条形图 (主图区域)
    main_ax_left = cbar_left + colorbar_width + 0.05
    ax0 = fig.add_axes([main_ax_left, plot_bottom, main_plot_width, plot_height])
    ax0.xaxis.tick_bottom()
    ax0.xaxis.set_label_position("bottom")
    ax0.invert_xaxis()

    # 绘制条形图，设置条形高度
    bar_height = 0.65
    bar_positions = range(len(sorted_features))
    bars = ax0.barh(y=bar_positions, width=sorted_shap_values, color=colors,
                    height=bar_height, edgecolor='white', linewidth=1.2)

    # 在每个条形上添加数值标签
    x_range = max(sorted_shap_values) - min(sorted_shap_values)
    offset = x_range * 0.01

    for i, (bar, value) in enumerate(zip(bars, sorted_shap_values)):
        label_x = value - offset
        label_y = i

        ax0.text(label_x, label_y, f'{value:.4f}',
                 ha='left', va='center',
                 fontsize=12, fontweight='bold', family='Arial',
                 bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                           edgecolor='#CCCCCC', alpha=0.9))

    ax0.invert_yaxis()
    ax0.set_xlabel('SHAP Value', size=28, labelpad=8, fontweight='bold', family='Arial')
    ax0.set_yticks([])
    ax0.spines[['left', 'top']].set_visible(False)
    ax0.spines['right'].set_position(('data', 0))
    ax0.spines['right'].set_visible(True)
    ax0.spines['bottom'].set_visible(True)
    ax0.tick_params(axis='x', which='major', direction='in', labelsize=18,
                    length=8, pad=10, width=2)
    for label in ax0.get_xticklabels():
        label.set_fontfamily('Arial')
        label.set_fontweight('bold')
    ax0.xaxis.set_minor_locator(ticker.AutoMinorLocator(10))
    ax0.tick_params(axis='x', which='minor', direction='in', length=5, width=1.5)
    for spine in ax0.spines.values():
        spine.set_linewidth(3)
        spine.set_color('#333333')

    # 玫瑰图 (嵌入条形图左下角)
    inset_size = min(main_plot_width, plot_height) * 0.65
    inset_left = main_ax_left - 0.08
    inset_bottom = plot_bottom - 0.02
    inset_ax_rect = [inset_left, inset_bottom, inset_size, inset_size]
    ax1 = fig.add_axes(inset_ax_rect, projection='polar')
    ax1.patch.set_alpha(0)
    ax1.bar(x=thetas, height=inner_heights, width=widths, color=inner_colors,
            align='edge', edgecolor='white', linewidth=2.0)
    ax1.bar(x=thetas, height=[colored_ring_width] * num_vars, width=widths,
            bottom=inner_heights, color=colors, align='edge', edgecolor='white',
            linewidth=2.0)
    for i in range(num_vars):
        label_angle_rad = thetas[i] + widths[i] / 2
        label_radius = total_lengths[i] + 0.6
        ax1.text(label_angle_rad, label_radius, f'{percentages[i]:.2f}%',
                 ha='center', va='center', fontsize=7, fontweight='bold', family='Arial',
                 bbox=dict(boxstyle='round,pad=0.15', facecolor='white',
                           edgecolor='#CCCCCC', alpha=0.9))
    ax1.set_yticklabels([])
    ax1.set_xticklabels([])
    ax1.spines['polar'].set_visible(False)
    ax1.grid(False)
    ax1.set_theta_zero_location('N')
    ax1.set_theta_direction(-1)
    ax1.set_ylim(0, max(total_lengths) + 2.5)

    # 特征名称 - 使用ax0的坐标系精确对齐条形位置
    label_fontsize = 24
    label_x = 1 - right_margin + 0.02

    for i, feature in enumerate(sorted_features):
        display_coords = ax0.transData.transform((0, i))
        fig_coords = fig.transFigure.inverted().transform(display_coords)
        y_position = fig_coords[1]

        fig.text(label_x, y_position, feature,
                 ha='left', va='center',
                 color='black', fontsize=label_fontsize, fontweight='bold', family='Arial')

    plt.tight_layout()
    output_file = os.path.join(OUTPUT_DIR, f'shap_barplot_rose_{model_name}_{color_scheme}.jpg')
    plt.savefig(output_file, dpi=600, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    return output_file


def plot_shap_beeswarm(shap_values, X_data, feature_names, model_name, color_scheme='viridis'):
    """
    绘制蜂窝图 (单独输出)
    """
    # 选择指定特征的数据
    feature_indices = [list(X_data.columns).index(f) for f in feature_names]
    shap_subset = shap_values[:, feature_indices]
    X_subset = X_data[feature_names]

    # 计算排序
    mean_abs_shap = np.abs(shap_subset).mean(axis=0)
    shap_series = pd.Series(mean_abs_shap, index=feature_names)
    shap_series.sort_values(ascending=False, inplace=True)
    sorted_features = shap_series.index.tolist()

    # 重排SHAP值以匹配排序
    sorted_indices = [feature_names.index(f) for f in sorted_features]
    shap_subset_sorted = shap_subset[:, sorted_indices]
    X_subset_sorted = X_subset[sorted_features]

    # 创建图形
    fig, ax = plt.subplots(figsize=(10, 10), dpi=600, facecolor='white')

    # 使用shap.summary_plot绘制
    shap.summary_plot(shap_subset_sorted, X_subset_sorted,
                      plot_type="dot", show=False,
                      max_display=len(sorted_features), cmap=color_scheme)

    # 设置标签和刻度
    ax = plt.gca()
    ax.set_xlabel("SHAP Value", fontsize=28, family='Arial', fontweight='bold', labelpad=12)
    ax.set_ylabel('')
    ax.tick_params(axis='x', labelsize=24, direction='in', width=2, length=8, pad=10)
    ax.tick_params(axis='y', labelsize=18, direction='out', width=2, length=6, pad=8)

    for label in ax.get_xticklabels():
        label.set_fontfamily('Arial')
        label.set_fontweight('bold')

    for label in ax.get_yticklabels():
        label.set_fontfamily('Arial')
        label.set_fontweight('bold')
        label.set_fontsize(22)

    ax.spines['bottom'].set_linewidth(3)
    ax.spines['bottom'].set_color('#333333')
    ax.spines['left'].set_linewidth(3)
    ax.spines['left'].set_color('#333333')
    for spine_name in ['top', 'right']:
        if spine_name in ax.spines:
            ax.spines[spine_name].set_visible(False)

    # 右侧颜色条设置
    if len(fig.axes) > 3:
        cbar_ax = fig.axes[-1]
        cbar_ax.set_ylabel('Feature Value', size=28, rotation=270,
                           labelpad=15, fontweight='bold', family='Arial')
        cbar_ax.tick_params(labelsize=24, width=2)
        tick_labels = cbar_ax.get_yticklabels()
        if len(tick_labels) >= 2:
            tick_labels[0].set_text("Low")
            tick_labels[-1].set_text("High")
            for tick_label in tick_labels:
                tick_label.set_fontweight('bold')
                tick_label.set_fontfamily('Arial')
            cbar_ax.set_yticklabels(tick_labels, fontsize=24, fontweight='bold', family='Arial')

    plt.tight_layout()
    output_file = os.path.join(OUTPUT_DIR, f'shap_beeswarm_{model_name}_{color_scheme}.jpg')
    plt.savefig(output_file, dpi=600, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    return output_file


def plot_shap_violin(shap_values, X_data, feature_names, model_name, color_scheme='viridis'):
    """
    绘制SHAP小提琴图 - 使用与蜂窝图一致的配色
    """
    # 选择指定特征的数据
    feature_indices = [list(X_data.columns).index(f) for f in feature_names]
    shap_subset = shap_values[:, feature_indices]
    X_subset = X_data[feature_names]

    # 计算排序
    mean_abs_shap = np.abs(shap_subset).mean(axis=0)
    shap_series = pd.Series(mean_abs_shap, index=feature_names)
    shap_series.sort_values(ascending=False, inplace=True)
    sorted_features = shap_series.index.tolist()

    # 重排SHAP值
    sorted_indices = [feature_names.index(f) for f in sorted_features]
    shap_subset_sorted = shap_subset[:, sorted_indices]
    X_subset_sorted = X_subset[sorted_features]

    # 创建图形
    fig, ax = plt.subplots(figsize=(20, 10), dpi=600, facecolor='white')

    # 使用指定配色方案绘制
    shap.summary_plot(shap_subset_sorted, X_subset_sorted,
                      plot_type="layered_violin", cmap=color_scheme,
                      show=False, max_display=len(feature_names))

    # 设置标签
    plt.xlabel('SHAP Value', fontsize=28, fontweight='bold', family='Arial', labelpad=12)

    # 获取当前坐标轴
    ax = plt.gca()

    # 将主Y轴（特征名称）移到右边
    ax.yaxis.tick_right()
    ax.yaxis.set_label_position("right")

    # 设置刻度
    ax.tick_params(axis='x', labelsize=24, width=2, length=6)
    ax.tick_params(axis='y', labelsize=24, width=2, length=6)

    for label in ax.get_xticklabels():
        label.set_fontfamily('Arial')
        label.set_fontweight('bold')

    for label in ax.get_yticklabels():
        label.set_fontfamily('Arial')
        label.set_fontweight('bold')
        label.set_fontsize(22)

    yticklabels = [label.get_text().lstrip('- ') for label in ax.get_yticklabels()]
    ax.set_yticklabels(yticklabels)

    # 获取当前x轴范围
    current_xlim = ax.get_xlim()
    ax.spines['left'].set_bounds(ax.get_ylim()[0], ax.get_ylim()[1])
    ax.spines['bottom'].set_bounds(current_xlim[0], current_xlim[1])

    # 设置边框
    for spine in ax.spines.values():
        spine.set_linewidth(2)

    # 调整颜色条
    if len(fig.axes) > 1:
        cbar_ax = fig.axes[-1]
        cbar_ax.set_ylabel('')

        cbar_pos = cbar_ax.get_position()
        new_pos = [0.08, cbar_pos.y0, cbar_pos.width, cbar_pos.height]
        cbar_ax.set_position(new_pos)

        main_ax_pos = ax.get_position()
        ax.set_position([0.15, main_ax_pos.y0, main_ax_pos.width - 0.05, main_ax_pos.height])

        cbar_ax.yaxis.set_ticks_position('left')
        cbar_ax.yaxis.set_label_position('left')

        cbar_ax.text(-3.5, 0.5, 'Feature Value', transform=cbar_ax.transAxes,
                     fontsize=28, rotation=90, va='center', fontweight='bold', ha='center', family='Arial')
        cbar_ax.set_facecolor('white')

        cbar_ax.tick_params(labelsize=24, width=2)
        for tick_label in cbar_ax.get_yticklabels():
            tick_label.set_fontweight('bold')
            tick_label.set_fontfamily('Arial')

    output_file = os.path.join(OUTPUT_DIR, f'shap_violin_{model_name}_{color_scheme}.jpg')
    plt.savefig(output_file, dpi=600, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    return output_file
def plot_shap_beeswarm_rose_combined(
        shap_values,
        X_data_df,
        importance_df,
        model_name,
        top_n=15,
        color_scheme='RdBu_r',
        output_dir=None
):
    """
    新增组合图：
    左侧：SHAP蜂窝图
    右侧：玫瑰图（SHAP贡献百分比）
    基于feng-mei-plot.py的绘图风格

    Parameters:
    -----------
    shap_values : array
        SHAP值矩阵 (仅描述符部分)
    X_data_df : DataFrame
        特征数据 (仅描述符部分)
    importance_df : DataFrame
        特征重要性DataFrame，包含'Feature'和'Mean_Abs_SHAP'列
    model_name : str
        模型名称
    top_n : int
        显示的Top特征数量
    color_scheme : str
        配色方案名称
    output_dir : str
        输出目录
    """

    print(f"\n正在生成 {model_name} 的蜂巢图 + 玫瑰图组合...")

    if output_dir is None:
        output_dir = OUTPUT_DIR

    # ===== 布局参数 (来自feng-mei-plot.py) =====
    SHAP_X = 0.05  # 蜂巢图左下角的 X 坐标
    SHAP_Y = 0.15  # 蜂巢图左下角的 Y 坐标
    SHAP_W = 0.55  # 蜂巢图的宽度
    SHAP_H = 0.75  # 蜂巢图的高度

    POLAR_X = 0.72  # 极坐标图左下角的 X 坐标
    POLAR_Y = 0.1  # 极坐标图左下角的 Y 坐标
    POLAR_SIZE = 0.75  # 极坐标图的整体尺寸
    POLAR_BOTTOM_VAL = 15  # 极坐标图内圈空心圆的半径
    POLAR_GAP = 2.0  # 内圈实线与柱子之间的间隔大小

    # ===== 获取Top特征 =====
    top_features = importance_df.sort_values(
        'Mean_Abs_SHAP',
        ascending=False
    ).head(top_n)

    feature_names = top_features['Feature'].tolist()
    mean_abs_shap = top_features['Mean_Abs_SHAP'].values
    # 按重要性降序排列（从下到上）
    feature_names_rev = feature_names[::-1]
    mean_abs_shap_rev = mean_abs_shap[::-1]
    # ===== 提取SHAP子集 =====
    feature_indices = [list(X_data_df.columns).index(f) for f in feature_names]
    shap_subset = shap_values[:, feature_indices]
    X_subset = X_data_df[feature_names]

    # ===== 计算玫瑰图数据 =====
    percentages = mean_abs_shap / mean_abs_shap.sum() * 100
    theta = np.linspace(0.0, 2 * np.pi, len(feature_names), endpoint=False)
    width = 2 * np.pi / len(feature_names)

    # 计算柱子宽度，缩小以创建间隔
    n_features = len(feature_names)
    total_angle = 2 * np.pi
    gap_ratio = 0.1  # 间隔占比（10%）

    # 每个柱子的角度宽度 = 总角度/特征数 * (1 - gap_ratio)
    width = (total_angle / n_features) * (1 - gap_ratio)

    # 计算每个柱子的起始角度（留出间隔）
    theta = np.linspace(0, total_angle, n_features, endpoint=False)
    # 添加半个间隔的偏移，使柱子居中
    theta = theta + width / 2 + (total_angle / n_features) * gap_ratio / 2

    # ===== 创建内圈圆数据 =====
    circle_theta = np.linspace(0, 2 * np.pi, 100)
    circle_r = np.full_like(circle_theta, POLAR_BOTTOM_VAL)

    # ===== 创建画布 =====
    fig = plt.figure(figsize=(40, 10), dpi=600, facecolor='white')

    # ===== 获取配色 =====
    if color_scheme in COLOR_SCHEMES:
        cmap = COLOR_SCHEMES[color_scheme]
    else:
        cmap = plt.cm.RdBu_r

    vmin = 0
    vmax = np.ceil(max(mean_abs_shap))

    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    colors = cmap(norm(mean_abs_shap))

    # ==================================================
    # 左侧：蜂窝图
    # ==================================================
    ax1_pos = [SHAP_X, SHAP_Y, SHAP_W, SHAP_H]
    ax1 = fig.add_axes(ax1_pos)
    plt.sca(ax1)

    shap.summary_plot(
        shap_subset,
        X_subset,
        plot_type="dot",
        show=False,
        max_display=len(feature_names),
        sort=False,
        cmap=cmap,
        color_bar=True
    )
    # 获取当前轴
    ax = plt.gca()

    # 遍历所有 PathCollection（scatter 点）
    for coll in ax.collections:
        coll.set_sizes([40])  # 40 = 点面积（默认大概16~20）

    fig = plt.gcf()
    axes = fig.axes
    cbar_ax = axes[-1]

    # ===============================
    # 统一坐标轴风格（核心修改）
    # ===============================

    # X轴标签
    ax1.set_xlabel(
        "SHAP Value",
        fontsize=28,
        fontweight='bold',
        family='Arial',
        labelpad=12
    )

    # 主刻度
    ax1.tick_params(
        axis='x',
        which='major',
        direction='in',
        labelsize=18,
        length=8,
        pad=10,
        width=2
    )

    # 次刻度
    import matplotlib.ticker as ticker
    ax1.xaxis.set_minor_locator(ticker.AutoMinorLocator(5))
    ax1.tick_params(
        axis='x',
        which='minor',
        direction='in',
        length=5,
        width=1.5
    )

    # Y轴刻度
    ax1.tick_params(
        axis='y',
        which='major',
        direction='in',
        labelsize=18,
        width=2
    )

    # 设置刻度字体
    for label in ax1.get_xticklabels():
        label.set_fontfamily('Arial')
        label.set_fontweight('bold')

    for label in ax1.get_yticklabels():
        label.set_fontfamily('Arial')
        label.set_fontweight('bold')

    # ===============================
    # 轴线加粗
    # ===============================
    for spine in ax1.spines.values():
        spine.set_linewidth(3)
        spine.set_color('#333333')

    # ===============================
    # colorbar风格统一
    # ===============================
    cbar_ax.set_ylabel(
        "Feature value",
        fontsize=16,
        fontweight='bold',
        family='Arial',
        rotation=90
    )

    cbar_ax.yaxis.set_label_coords(2.0, 0.5)

    cbar_ax.tick_params(
        labelsize=14,
        width=2,
        length=6,
        direction='in'
    )

    for label in cbar_ax.get_yticklabels():
        label.set_fontfamily('Arial')
        label.set_fontweight('bold')
    # ==================================================
    # 右侧：玫瑰图
    # ==================================================
    ax2_pos = [POLAR_X, POLAR_Y, POLAR_SIZE, POLAR_SIZE]
    ax2 = fig.add_axes(ax2_pos, projection='polar')

    # 归一化颜色
    norm = mcolors.Normalize(vmin=min(mean_abs_shap), vmax=max(mean_abs_shap))
    colors = cmap(norm(mean_abs_shap))

    # 绘制极坐标柱状图
    bars = ax2.bar(
        theta,
        percentages,
        width=width,
        bottom=POLAR_BOTTOM_VAL + POLAR_GAP,
        color=colors,
        edgecolor='black',
        linewidth=0.8
    )

    # 绘制内圈圆
    ax2.plot(circle_theta, circle_r,
             color='black',
             linewidth=1,
             linestyle='-')

    ax2.set_theta_zero_location("N")  # 极坐标的零度方向为正北
    ax2.set_theta_direction(-1)  # 极坐标的角度增长方向为顺时针
    ax2.set_axis_off()  # 关闭默认的坐标轴


    # 添加特征标签
    for angle, percent, name, raw_val in zip(theta, percentages, feature_names, mean_abs_shap):
        angle_deg = np.degrees(angle)  # 将弧度转换为角度

        # 计算半径 (基础 + 间隔 + 柱子长度)
        visual_top = POLAR_BOTTOM_VAL + POLAR_GAP + percent

        # 根据角度确定标签位置和旋转
        if 0 <= angle_deg < 180:
            # 右半圆
            rotation = 90 - angle_deg
            alignment_ha = 'left'
            alignment_va = 'center'
            pos_outer = visual_top+0.5 # 标签放在柱子外侧
        else:
            # 左半圆
            rotation = 270 - angle_deg
            alignment_ha = 'right'
            alignment_va = 'center'
            pos_outer = visual_top+0.5  # 标签放在柱子外侧

        # 添加带白色背景的标签
        ax2.text(angle,
                 pos_outer,
                 f"{name}\n{percent:.1f}%",
                 ha=alignment_ha,
                 va=alignment_va,
                 rotation=rotation,
                 rotation_mode='anchor',
                 fontsize=14,
                 fontweight='bold',
                 family='Arial',
                 color='black',
                 bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8, edgecolor="none"))

    for angle, percent, raw_val in zip(theta, percentages, mean_abs_shap):

        angle_deg = np.degrees(angle)
        visual_top = POLAR_BOTTOM_VAL + POLAR_GAP + percent

        # 文字放在柱子内部顶部（按比例偏移，更自然）
        text_radius = visual_top - percent * 0.12

        # 控制旋转方向（保证文字始终朝外）
        if 0 <= angle_deg < 180:
            rotation = - angle_deg
        else:
            rotation = 180 - angle_deg

        ax2.text(
            angle,
            text_radius,
            f"{raw_val:.3f}",  # 显示 mean_abs_shap
            ha='center',
            va='center',
            rotation=rotation,
            rotation_mode='anchor',
            fontsize=9,
            fontweight='bold',
            color='white'
        )
    # 添加颜色条
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cax = ax2.inset_axes([0.48,0.35,0.04,0.3], transform = ax2.transAxes)
    cbar = plt.colorbar(sm, cax=cax)
    cbar.set_label('SHAP Value', fontsize=10, fontweight='bold', family='Arial',rotation=0)
    cbar.ax.yaxis.set_label_coords(0.5, 1.1)
    cbar.ax.yaxis.set_ticks_position('right')
    cbar.ax.tick_params(labelsize=12)
    for label in cbar.ax.get_yticklabels():
        label.set_fontfamily('Arial')
        label.set_fontweight('bold')



    # 保存
    output_path = os.path.join(output_dir, f"shap_beeswarm_rose_{model_name}_{color_scheme}.jpg")
    output_path_pdf = os.path.join(output_dir, f"shap_beeswarm_rose_{model_name}_{color_scheme}.pdf")

    plt.savefig(output_path, dpi=600, bbox_inches='tight', facecolor='white')
    plt.savefig(output_path_pdf, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"  ✓ 已保存: {os.path.basename(output_path)}")
    return output_path


# ==========================================
# 9. 生成所有SHAP可视化 (修改这部分，添加新组合图)
# ==========================================
print("\n" + "=" * 80)
print("步骤 5: 生成SHAP可视化")
print("=" * 80)

models_to_plot = [
    ('LightGBM', lightgbm_shap_values, lightgbm_importance, lightgbm_top_features),
    ('CatBoost', catboost_shap_values, catboost_importance, catboost_top_features),
    ('XGBoost', xgboost_shap_values, xgboost_importance, xgboost_top_features),
    ('RandomForest', randomforest_shap_values, randomforest_importance, randomforest_top_features),
    ('GradientBoosting', gradient_shap_values, gradient_importance, gradient_top_features)
]

generated_files = []

for model_name, shap_vals, importance_df, top_features in models_to_plot:
    print(f"\n{model_name} 模型可视化:")

    # 1. 生成条形图+玫瑰图组合 (您原有的)
    print(f"  生成条形图+玫瑰图组合...")
    for scheme_name in COLOR_SCHEMES.keys():
        output_file = plot_shap_barplot_with_rose(shap_vals, X_test_desc_df, top_features,
                                                  model_name, scheme_name)
        generated_files.append(output_file)
        print(f"    ✓ {scheme_name}: {os.path.basename(output_file)}")

    # 2. 生成蜂窝图 (单独) - 您原有的
    print(f"  生成蜂窝图...")
    for scheme_name in COLOR_SCHEMES.keys():
        output_file = plot_shap_beeswarm(shap_vals, X_test_desc_df, top_features,
                                         model_name, scheme_name)
        generated_files.append(output_file)
        print(f"    ✓ {scheme_name}: {os.path.basename(output_file)}")

    # 3. 生成小提琴图 - 您原有的
    print(f"  生成小提琴图...")
    for scheme_name in COLOR_SCHEMES.keys():
        output_file = plot_shap_violin(shap_vals, X_test_desc_df, top_features,
                                       model_name, scheme_name)
        generated_files.append(output_file)
        print(f"    ✓ {scheme_name}: {os.path.basename(output_file)}")

    # 4. 新增：生成蜂巢图+玫瑰图组合
    print(f"  生成蜂巢图+玫瑰图组合 (新增)...")
    for scheme_name in COLOR_SCHEMES.keys():
        output_file = plot_shap_beeswarm_rose_combined(
            shap_vals,
            X_test_desc_df,
            importance_df,  # 使用完整的importance_df而不是仅top_features
            model_name,
            top_n=TOP_N_FEATURES,
            color_scheme=scheme_name,
            output_dir=OUTPUT_DIR
        )
        generated_files.append(output_file)
        print(f"    ✓ {scheme_name}: {os.path.basename(output_file)}")


# ==========================================
# 10. 批量生成所有模型的组合图（可选：只生成一种配色）
# ==========================================
def generate_all_combined_plots_fast():
    """
    快速生成所有模型的蜂巢图+玫瑰图组合（只使用RdBu_r配色）
    """
    print("\n" + "=" * 80)
    print("快速生成所有模型的蜂巢图+玫瑰图组合")
    print("=" * 80)

    models_data = [
        ('LightGBM', lightgbm_shap_values, lightgbm_importance),
        ('CatBoost', catboost_shap_values, catboost_importance),
        ('XGBoost', xgboost_shap_values, xgboost_importance),
        ('RandomForest', randomforest_shap_values, randomforest_importance),
        ('GradientBoosting', gradient_shap_values, gradient_importance)
    ]

    for model_name, shap_vals, importance_df in models_data:
        plot_shap_beeswarm_rose_combined(
            shap_vals,
            X_test_desc_df,
            importance_df,
            model_name,
            top_n=TOP_N_FEATURES,
            color_scheme='RdBu_r',  # 使用默认配色
            output_dir=OUTPUT_DIR
        )

    print("\n✓ 所有模型的蜂巢图+玫瑰图组合已生成完成")


# 如果需要快速生成（只使用一种配色），取消下面的注释
# generate_all_combined_plots_fast()

# ==========================================
# 11. 生成汇总报告（需要更新文件计数）
# ==========================================
print("\n" + "=" * 80)
print("步骤 6: 生成分析报告")
print("=" * 80)

# 计算文件数量
n_models = len(models_to_plot)
n_schemes = len(COLOR_SCHEMES)
n_bar_rose = n_models * n_schemes  # 条形图+玫瑰图
n_beeswarm = n_models * n_schemes  # 蜂窝图单独
n_violin = n_models * n_schemes  # 小提琴图
n_beeswarm_rose = n_models * n_schemes  # 新增的蜂巢图+玫瑰图组合
total_viz = n_bar_rose + n_beeswarm + n_violin + n_beeswarm_rose

summary_report = f"""
{'=' * 80}
SHAP分析报告 - 从已保存模型加载
数据文件: sys1.csv
{'=' * 80}

数据集信息:
- 原始样本数: {len(df)}
- 有效样本数: {len(X)}
- 训练集: {len(X_train)} 条 (85%)
- 测试集: {len(X_test)} 条 (15%)
- 总特征维度: {X.shape[1]} (指纹: 3072, 描述符: {n_desc})
- 分析的描述符数: {X_test_desc.shape[1]}
- 选取Top特征数: {TOP_N_FEATURES}
- 数据划分种子: {best_seed}

模型性能 (测试集):
{'=' * 80}
1. LightGBM:
   - R² Score: {lightgbm_r2_test:.4f}
   - RMSE: {lightgbm_rmse_test:.4f}
   - MAE: {lightgbm_mae_test:.4f}

2. CatBoost:
   - R² Score: {catboost_r2_test:.4f}
   - RMSE: {catboost_rmse_test:.4f}
   - MAE: {catboost_mae_test:.4f}

3. XGBoost:
   - R² Score: {xgboost_r2_test:.4f}
   - RMSE: {xgboost_rmse_test:.4f}
   - MAE: {xgboost_mae_test:.4f}

4. RandomForest:
   - R² Score: {randomforest_r2_test:.4f}
   - RMSE: {randomforest_rmse_test:.4f}
   - MAE: {randomforest_mae_test:.4f}

5. GradientBoosting:
   - R² Score: {gradient_r2_test:.4f}
   - RMSE: {gradient_rmse_test:.4f}
   - MAE: {gradient_mae_test:.4f}

LightGBM Top {TOP_N_FEATURES} 描述符 (按SHAP重要性排序):
{'=' * 80}
{chr(10).join([f"{i + 1:2d}. {f:30s} - SHAP值: {lightgbm_importance[lightgbm_importance['Feature'] == f]['Mean_Abs_SHAP'].values[0]:.6f}"
               for i, f in enumerate(lightgbm_top_features)])}

CatBoost Top {TOP_N_FEATURES} 描述符 (按SHAP重要性排序):
{'=' * 80}
{chr(10).join([f"{i + 1:2d}. {f:30s} - SHAP值: {catboost_importance[catboost_importance['Feature'] == f]['Mean_Abs_SHAP'].values[0]:.6f}"
               for i, f in enumerate(catboost_top_features)])}

输出文件:
{'=' * 80}
数据文件:
- feature_importance_all.csv: 所有描述符的SHAP重要性
- top{TOP_N_FEATURES}_features.csv: Top {TOP_N_FEATURES} 描述符对比
- model_performance.csv: 模型性能指标
- test_predictions.csv: 测试集预测结果

可视化文件 (每个模型x每个配色方案):
- 条形图+玫瑰图组合: {n_bar_rose}个文件
- 蜂窝图(单独): {n_beeswarm}个文件
- 小提琴图: {n_violin}个文件
- 蜂巢图+玫瑰图组合(新增): {n_beeswarm_rose}个文件
- 总共: {total_viz}个可视化文件

配色方案说明:
{'=' * 80}
- viridis: 绿-黄渐变 (高对比度,色盲友好)
- plasma: 紫-橙渐变 (鲜艳,高可见性)
- coolwarm: 蓝-红渐变 (经典科学配色)
- RdYlBu: 红-黄-蓝 (三色渐变)
- RdBu_r: 红-蓝反向 (双色对比)

输出目录: {OUTPUT_DIR}
生成时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
{'=' * 80}
"""

report_path = os.path.join(OUTPUT_DIR, "SHAP_Analysis_Report.txt")
with open(report_path, 'w', encoding='utf-8') as f:
    f.write(summary_report)
print(f"✓ 分析报告已保存: {report_path}")

print("\n" + summary_report)

print("\n" + "=" * 80)
print("✅ 所有任务完成!")
print("=" * 80)
print(f"\n📁 所有文件已保存到: {OUTPUT_DIR}")
print(f"\n📊 共生成 {total_viz + 4} 个文件:")
print(f"   - {total_viz} 个可视化图表")
print(f"   - 4 个数据/报告文件")
print("\n📊 可视化文件分类:")
print(f"   - 条形图+玫瑰图组合: {n_bar_rose}个")
print(f"   - 蜂窝图(单独): {n_beeswarm}个")
print(f"   - 小提琴图: {n_violin}个")
print(f"   - 蜂巢图+玫瑰图组合(新增): {n_beeswarm_rose}个")
print("\n" + "=" * 80)