"""
t-SNE 和 UMAP 可视化脚本
以 4567.csv 为基础数据,用 412.csv 中的结合能着色

运行前请先安装依赖:
pip install pandas numpy matplotlib scikit-learn umap-learn
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

# 尝试导入umap
try:
    from umap import UMAP
except ImportError:
    print("请先安装 umap-learn: pip install umap-learn")
    exit(1)

import warnings
warnings.filterwarnings('ignore')

# ========== 字体设置 ==========
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['axes.unicode_minus'] = False
# ==============================

# ========== 配置区域 ==========
# 修改为您本地的文件路径
BASE_FILE = '4599.csv'      # 大数据集
BINDING_FILE = '412.csv'    # 含结合能的数据集
OUTPUT_FILE = 'tsne_umap_binding_energy.png'

# 输出文件名
TSNE_UMAP_CSV = 'tsne_umap_results.csv'  # 完整结果（包含所有t-SNE和UMAP值）
BINDING_412_CSV = '412_tsne_umap.csv'     # 只包含412个有结合能分子的降维结果
NEAR_ZERO_CSV = 'binding_near_0.csv'
NEAR_ONE_CSV = 'binding_near_1.csv'

# 定义"接近"的阈值
THRESHOLD_ZERO = 0.1  # 结合能接近0的阈值: [0, 0.1]

# 字体大小设置
TICK_FONTSIZE = 22    # 坐标轴刻度字体大小
LABEL_FONTSIZE = 26   # 标签字体大小

# 点的大小设置
POINT_SIZE_LARGE = 80   # 有结合能的点的大小(增大)
POINT_SIZE_SMALL = 25   # 无结合能的点的大小(适当增大)
# ==============================

# 读取数据
print("读取数据...")
df_base = pd.read_csv(BASE_FILE)
df_binding = pd.read_csv(BINDING_FILE)

print(f"4567.csv: {df_base.shape[0]} 条记录")
print(f"412.csv: {df_binding.shape[0]} 条记录")

# 筛选成功的分子
df_base = df_base[df_base['Status'] == 'Success'].copy()
df_binding = df_binding[df_binding['Status'] == 'Success'].copy()

print(f"筛选后 4567.csv: {len(df_base)} 个分子")
print(f"筛选后 412.csv: {len(df_binding)} 个分子")

# 准备特征 (使用 MolWt, LogP, TPSA 作为分子描述符)
features = ['MolWt', 'LogP', 'TPSA']
X = df_base[features].values

# 标准化特征
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 通过 SMILES 匹配结合能数据
binding_dict = dict(zip(df_binding['SMILES'], df_binding['HA_binding energy']))
df_base['binding_energy'] = df_base['SMILES'].map(binding_dict)

# 统计匹配情况
matched = df_base['binding_energy'].notna().sum()
print(f"匹配到结合能的分子数: {matched}")

# 运行 t-SNE
print("\n正在运行 t-SNE (可能需要几分钟)...")
tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
X_tsne = tsne.fit_transform(X_scaled)

# 运行 UMAP
print("正在运行 UMAP...")
reducer = UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
X_umap = reducer.fit_transform(X_scaled)

# ========== 保存降维结果到CSV ==========
print("\n保存降维结果...")

# 添加t-SNE和UMAP的列到数据框
df_base['tsne_1'] = X_tsne[:, 0]
df_base['tsne_2'] = X_tsne[:, 1]
df_base['umap_1'] = X_umap[:, 0]
df_base['umap_2'] = X_umap[:, 1]

# 调整列的顺序，使降维结果更容易识别
# 将t-SNE和UMAP的列放在前面
cols = df_base.columns.tolist()
# 移除降维相关的列
other_cols = [col for col in cols if col not in ['tsne_1', 'tsne_2', 'umap_1', 'umap_2']]
# 重新排列：降维列在前，其他列在后
new_cols = ['tsne_1', 'tsne_2', 'umap_1', 'umap_2'] + other_cols
df_base_reordered = df_base[new_cols]

# 保存完整结果（所有分子的t-SNE和UMAP坐标）
df_base_reordered.to_csv(TSNE_UMAP_CSV, index=False)
print(f"✓ 已保存完整降维结果到: {TSNE_UMAP_CSV}")
print(f"  - 包含 {len(df_base)} 个分子的完整t-SNE和UMAP坐标")
print(f"  - 列: tsne_1, tsne_2, umap_1, umap_2, 以及其他原始数据列")

# ========== 单独保存412个有结合能分子的降维结果 ==========
df_with_binding = df_base[df_base['binding_energy'].notna()].copy()
if len(df_with_binding) > 0:
    df_with_binding_reordered = df_with_binding[new_cols]
    df_with_binding_reordered.to_csv(BINDING_412_CSV, index=False)
    print(f"✓ 已保存 {len(df_with_binding)} 个有结合能分子的降维结果到: {BINDING_412_CSV}")
    print(f"  - 包含这些分子的t-SNE和UMAP坐标及结合能数据")
else:
    print(f"✗ 没有找到有结合能的分子")

# ========== 筛选并保存接近0和1的点 ==========
# 接近0的点: binding_energy <= THRESHOLD_ZERO
df_near_zero = df_with_binding[df_with_binding['binding_energy'] <= THRESHOLD_ZERO]
if len(df_near_zero) > 0:
    # 同样调整列顺序
    df_near_zero_reordered = df_near_zero[new_cols]
    df_near_zero_reordered.to_csv(NEAR_ZERO_CSV, index=False)
    print(f"✓ 已保存 {len(df_near_zero)} 个结合能接近0的点到: {NEAR_ZERO_CSV}")
else:
    print(f"✗ 没有找到结合能 <= {THRESHOLD_ZERO} 的点")

# 接近1的点: binding_energy >= 0.85
df_near_one = df_with_binding[df_with_binding['binding_energy'] >= 0.85]
if len(df_near_one) > 0:
    # 同样调整列顺序
    df_near_one_reordered = df_near_one[new_cols]
    df_near_one_reordered.to_csv(NEAR_ONE_CSV, index=False)
    print(f"✓ 已保存 {len(df_near_one)} 个结合能 >= 0.85 的点到: {NEAR_ONE_CSV}")
else:
    print(f"✗ 没有找到结合能 >= 0.85 的点")

# 创建图形 - 两个正方形子图
fig, axes = plt.subplots(1, 2, figsize=(20, 9))

# 获取结合能数据
binding_energy = df_base['binding_energy'].values
has_binding = ~np.isnan(binding_energy)

# 颜色映射 (红-黄-蓝反转,高值为红色)
cmap = plt.cm.RdYlBu_r

# 对有结合能的点按结合能值排序,让高值点绘制在上层
binding_indices = np.where(has_binding)[0]
binding_values = binding_energy[has_binding]
sorted_order = np.argsort(binding_values)  # 从低到高排序
sorted_indices = binding_indices[sorted_order]

# ===== 绘制 t-SNE 图 =====
ax1 = axes[0]
ax1.set_box_aspect(1)  # 设置正方形框架

# 先绘制没有结合能数据的点(灰色,适当增大)
ax1.scatter(X_tsne[~has_binding, 0], X_tsne[~has_binding, 1],
            c='lightgray', s=POINT_SIZE_SMALL, alpha=0.3, label='_nolegend_')
# 再按结合能从低到高绘制有结合能数据的点,使高值点在上层
scatter1 = ax1.scatter(X_tsne[sorted_indices, 0], X_tsne[sorted_indices, 1],
                       c=binding_energy[sorted_indices], cmap=cmap, s=POINT_SIZE_LARGE, alpha=0.8,
                       edgecolors='black', linewidths=0.3)
cbar1 = plt.colorbar(scatter1, ax=ax1, shrink=0.8)
cbar1.set_label('HA Binding Energy', fontsize=LABEL_FONTSIZE, fontname='Arial', fontweight='bold')
cbar1.ax.tick_params(labelsize=TICK_FONTSIZE)
# 设置colorbar刻度标签加粗
for label in cbar1.ax.get_yticklabels():
    label.set_fontweight('bold')
    label.set_fontname('Arial')

ax1.set_xlabel('t-SNE 1', fontsize=LABEL_FONTSIZE, fontname='Arial', fontweight='bold')
ax1.set_ylabel('t-SNE 2', fontsize=LABEL_FONTSIZE, fontname='Arial', fontweight='bold')
ax1.set_title('t-SNE', fontsize=LABEL_FONTSIZE, fontname='Arial', fontweight='bold')
ax1.tick_params(axis='both', labelsize=TICK_FONTSIZE)
# 设置坐标轴刻度标签加粗
for label in ax1.get_xticklabels() + ax1.get_yticklabels():
    label.set_fontweight('bold')
    label.set_fontname('Arial')
ax1.legend(loc='upper right', fontsize=16, prop={'family': 'Arial', 'weight': 'bold'})

# ===== 绘制 UMAP 图 =====
ax2 = axes[1]
ax2.set_box_aspect(1)  # 设置正方形框架

# 先绘制没有结合能数据的点(灰色,适当增大)
ax2.scatter(X_umap[~has_binding, 0], X_umap[~has_binding, 1],
            c='lightgray', s=POINT_SIZE_SMALL, alpha=0.3, label='_nolegend_')
# 再按结合能从低到高绘制有结合能数据的点,使高值点在上层
scatter2 = ax2.scatter(X_umap[sorted_indices, 0], X_umap[sorted_indices, 1],
                       c=binding_energy[sorted_indices], cmap=cmap, s=POINT_SIZE_LARGE, alpha=0.8,
                       edgecolors='black', linewidths=0.3)
cbar2 = plt.colorbar(scatter2, ax=ax2, shrink=0.8)
cbar2.set_label('HA Binding Energy', fontsize=LABEL_FONTSIZE, fontname='Arial', fontweight='bold')
cbar2.ax.tick_params(labelsize=TICK_FONTSIZE)
# 设置colorbar刻度标签加粗
for label in cbar2.ax.get_yticklabels():
    label.set_fontweight('bold')
    label.set_fontname('Arial')

ax2.set_xlabel('UMAP 1', fontsize=LABEL_FONTSIZE, fontname='Arial', fontweight='bold')
ax2.set_ylabel('UMAP 2', fontsize=LABEL_FONTSIZE, fontname='Arial', fontweight='bold')
ax2.set_title('UMAP', fontsize=LABEL_FONTSIZE, fontname='Arial', fontweight='bold')
ax2.tick_params(axis='both', labelsize=TICK_FONTSIZE)
# 设置坐标轴刻度标签加粗
for label in ax2.get_xticklabels() + ax2.get_yticklabels():
    label.set_fontweight('bold')
    label.set_fontname('Arial')
ax2.legend(loc='upper right', fontsize=16, prop={'family': 'Arial', 'weight': 'bold'})

plt.tight_layout()
plt.savefig(OUTPUT_FILE, dpi=600, bbox_inches='tight', facecolor='white', edgecolor='none')
print(f"\n✓ 图片已保存: {OUTPUT_FILE}")

# 打印统计信息
print(f"\n{'='*50}")
print(f"统计信息")
print(f"{'='*50}")
print(f"基础数据集分子数: {len(df_base)}")
print(f"有结合能的分子数: {matched}")
print(f"结合能范围: {binding_energy[has_binding].min():.4f} - {binding_energy[has_binding].max():.4f}")
print(f"结合能均值: {binding_energy[has_binding].mean():.4f}")
print(f"\n接近0 (≤{THRESHOLD_ZERO}) 的分子数: {len(df_near_zero)}")
print(f"结合能 >= 0.85 的分子数: {len(df_near_one)}")
print(f"{'='*50}")
print(f"\n输出文件说明:")
print(f"1. {TSNE_UMAP_CSV} - 包含所有{len(df_base)}个分子的完整降维坐标")
print(f"   列: tsne_1, tsne_2, umap_1, umap_2, [其他原始数据]")
print(f"2. {BINDING_412_CSV} - 只包含{len(df_with_binding)}个有结合能分子的降维坐标 ⭐新增")
print(f"   列: tsne_1, tsne_2, umap_1, umap_2, binding_energy, [其他数据]")
print(f"3. {NEAR_ZERO_CSV} - 结合能接近0的分子及其降维坐标")
print(f"4. {NEAR_ONE_CSV} - 结合能>=0.85的分子及其降维坐标")
print(f"5. {OUTPUT_FILE} - t-SNE和UMAP可视化图片")
print(f"{'='*50}")

plt.show()