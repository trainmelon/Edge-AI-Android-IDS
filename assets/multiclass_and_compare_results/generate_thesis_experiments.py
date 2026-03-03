import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc
import matplotlib
matplotlib.use('Agg') # 强制使用非交互式后端
import matplotlib.pyplot as plt
import seaborn as sns
import os

# 设置绘图风格
plt.style.use('ggplot')
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签，如果报错可注释掉
plt.rcParams['axes.unicode_minus'] = False

print("========== 1. 加载并预处理 NSL-KDD 数据集 ==========")
col_names = ["duration", "protocol_type", "service", "flag", "src_bytes",
             "dst_bytes", "land", "wrong_fragment", "urgent", "hot", "num_failed_logins",
             "logged_in", "num_compromised", "root_shell", "su_attempted", "num_root",
             "num_file_creations", "num_shells", "num_access_files", "num_outbound_cmds",
             "is_host_login", "is_guest_login", "count", "srv_count", "serror_rate",
             "srv_serror_rate", "rerror_rate", "srv_rerror_rate", "same_srv_rate",
             "diff_srv_rate", "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
             "dst_host_same_srv_rate", "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
             "dst_host_srv_diff_host_rate", "dst_host_serror_rate", "dst_host_srv_serror_rate",
             "dst_host_rerror_rate", "dst_host_srv_rerror_rate", "label", "difficulty_level"]

df = pd.read_csv('KDDTrain+.txt', header=None, names=col_names)
df.drop(['difficulty_level'], axis=1, inplace=True)
df['label'] = df['label'].apply(lambda x: 0 if x == 'normal' else 1)

for col in ['protocol_type', 'service', 'flag']:
    df[col] = LabelEncoder().fit_transform(df[col])

X = df.drop('label', axis=1).values
y = df['label'].values

scaler = StandardScaler()
X = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("========== 2. 开展横向对比实验 (多种算法 PK) ==========")
# 定义需要对比的传统机器学习模型
models = {
    "逻辑回归 (LR)": LogisticRegression(max_iter=1000),
    "决策树 (DT)": DecisionTreeClassifier(),
    "随机森林 (RF)": RandomForestClassifier(n_estimators=20, max_depth=10)
}

results = {}
roc_data = {}

# 训练传统模型并记录指标
for name, model in models.items():
    print(f"正在训练 {name}...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    results[name] = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1-Score": f1_score(y_test, y_pred)
    }
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_data[name] = (fpr, tpr, auc(fpr, tpr))

# 训练我们真正使用的深度学习模型 (轻量级 MLP)
print("正在训练 本文提出模型 (Lightweight MLP)...")
mlp_model = tf.keras.Sequential([
    tf.keras.layers.Dense(32, activation='relu', input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(2, activation='softmax')
])
mlp_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
mlp_model.fit(X_train, y_train, epochs=5, batch_size=256, verbose=0)  # 静默训练

y_prob_mlp = mlp_model.predict(X_test)[:, 1]
y_pred_mlp = np.argmax(mlp_model.predict(X_test), axis=1)

results["本文模型 (MLP)"] = {
    "Accuracy": accuracy_score(y_test, y_pred_mlp),
    "Precision": precision_score(y_test, y_pred_mlp),
    "Recall": recall_score(y_test, y_pred_mlp),
    "F1-Score": f1_score(y_test, y_pred_mlp)
}
fpr, tpr, _ = roc_curve(y_test, y_prob_mlp)
roc_data["本文模型 (MLP)"] = (fpr, tpr, auc(fpr, tpr))

# 打印最终数据表 (让同学直接复制进论文的表格里)
print("\n========== 核心评估指标对比表 ==========")
results_df = pd.DataFrame(results).T
print(results_df.round(4))

print("\n========== 3. 绘制并保存对比图表 ==========")
# 图1：综合指标柱状图
results_df.plot(kind='bar', figsize=(10, 6), colormap='viridis')
plt.title('不同算法在入侵检测上的性能对比')
plt.ylabel('Score (0.0 - 1.0)')
plt.xticks(rotation=0)
plt.legend(loc='lower right')
plt.tight_layout()
plt.savefig('实验图1_算法性能对比.png', dpi=300)

# 图2：ROC曲线图
plt.figure(figsize=(8, 6))
for name, (fpr, tpr, roc_auc) in roc_data.items():
    plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('假正率 (False Positive Rate)')
plt.ylabel('真正率 (True Positive Rate)')
plt.title('各检测模型的 ROC 曲线对比')
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig('实验图2_ROC曲线对比.png', dpi=300)
print("图表生成完毕！已保存为 '实验图1_算法性能对比.png' 和 '实验图2_ROC曲线对比.png'")

print("\n========== 4. 移动端轻量化消融实验 (核心工作量) ==========")
# 导出未量化模型
converter_normal = tf.lite.TFLiteConverter.from_keras_model(mlp_model)
tflite_normal = converter_normal.convert()
with open('Model_Float32.tflite', 'wb') as f:
    f.write(tflite_normal)

# 导出量化模型
converter_quant = tf.lite.TFLiteConverter.from_keras_model(mlp_model)
converter_quant.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_quant = converter_quant.convert()
with open('Model_Int8.tflite', 'wb') as f:
    f.write(tflite_quant)

# 对比体积大小
size_normal = os.path.getsize('Model_Float32.tflite') / 1024
size_quant = os.path.getsize('Model_Int8.tflite') / 1024
print(f"原始深度学习模型 (Float32) 体积: {size_normal:.2f} KB")
print(f"量化压缩后模型 (Int8) 体积:     {size_quant:.2f} KB")
print(f"✅ 模型体积成功压缩了 {(1 - size_quant / size_normal) * 100:.2f}%！")
print("提示：请将以上压缩比例数据直接写入论文『模型优化与固化』章节。")