import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib

matplotlib.use('Agg')  # 依然使用无头模式，防止画图弹窗报错
import matplotlib.pyplot as plt
import seaborn as sns
import os

# 设置中文字体（若你的电脑没装黑体，图表上的中文可能会显示方块，但不影响运行）
plt.style.use('ggplot')
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

print("========== 1. 加载并进行【5大细粒度分类】预处理 ==========")
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

# 核心秘籍：将 NSL-KDD 中 40 多种奇奇怪怪的攻击，映射为学术界标准的 5 大类
# 0: Normal(正常), 1: DoS(拒绝服务), 2: Probe(探测嗅探), 3: R2L(非法提权), 4: U2R(非法访问)
attack_map = {
    'normal': 0,
    'back': 1, 'land': 1, 'neptune': 1, 'pod': 1, 'smurf': 1, 'teardrop': 1, 'apache2': 1, 'udpstorm': 1,
    'processtable': 1, 'worm': 1,
    'satan': 2, 'ipsweep': 2, 'nmap': 2, 'portsweep': 2, 'mscan': 2, 'saint': 2,
    'guess_passwd': 3, 'ftp_write': 3, 'imap': 3, 'phf': 3, 'multihop': 3, 'warezmaster': 3, 'warezclient': 3, 'spy': 3,
    'xlock': 3, 'xsnoop': 3, 'snmpguess': 3, 'snmpgetattack': 3, 'httptunnel': 3, 'sendmail': 3, 'named': 3,
    'buffer_overflow': 4, 'loadmodule': 4, 'rootkit': 4, 'perl': 4, 'sqlattack': 4, 'xterm': 4, 'ps': 4
}
# 应用映射，未在字典里的罕见攻击默认归为 1 (DoS)
df['label'] = df['label'].apply(lambda x: attack_map.get(x, 1))
class_names = ['正常流量', 'DoS攻击', '探测嗅探', 'R2L越权', 'U2R提权']

for col in ['protocol_type', 'service', 'flag']:
    df[col] = LabelEncoder().fit_transform(df[col])

X = df.drop('label', axis=1).values
y = df['label'].values

scaler = StandardScaler()
X = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("========== 2. 开展多分类横向对比实验 ==========")
models = {
    "逻辑回归 (LR)": LogisticRegression(max_iter=2000),
    "决策树 (DT)": DecisionTreeClassifier(),
    "随机森林 (RF)": RandomForestClassifier(n_estimators=20, max_depth=15)
}

results = {}

# 多分类评估需要用 macro 平均
for name, model in models.items():
    print(f"正在训练 {name}...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    results[name] = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, average='macro', zero_division=0),
        "Recall": recall_score(y_test, y_pred, average='macro', zero_division=0),
        "F1-Score": f1_score(y_test, y_pred, average='macro', zero_division=0)
    }

print("正在训练 本文提出模型 (Lightweight MLP)...")
mlp_model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(32, activation='relu'),
    # 注意：输出层变成了 5 个神经元，对应 5 个分类！
    tf.keras.layers.Dense(5, activation='softmax')
])
mlp_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
mlp_model.fit(X_train, y_train, epochs=10, batch_size=256, verbose=0)

y_pred_mlp = np.argmax(mlp_model.predict(X_test), axis=1)

results["本文模型 (MLP)"] = {
    "Accuracy": accuracy_score(y_test, y_pred_mlp),
    "Precision": precision_score(y_test, y_pred_mlp, average='macro', zero_division=0),
    "Recall": recall_score(y_test, y_pred_mlp, average='macro', zero_division=0),
    "F1-Score": f1_score(y_test, y_pred_mlp, average='macro', zero_division=0)
}

print("\n========== 核心多分类评估指标对比表 ==========")
results_df = pd.DataFrame(results).T
print(results_df.round(4))

print("\n========== 3. 绘制并保存多分类对比图表 ==========")
# 图1：综合指标柱状图
results_df.plot(kind='bar', figsize=(10, 6), colormap='plasma')
plt.title('多分类任务下不同算法的性能对比')
plt.ylabel('Score (Macro Average)')
plt.xticks(rotation=0)
plt.legend(loc='lower right')
plt.tight_layout()
plt.savefig('实验图1_多分类性能对比.png', dpi=300)

# 图2：极其高级的“混淆矩阵热力图” (专门放进论文第四章)
cm = confusion_matrix(y_test, y_pred_mlp)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.xlabel('预测类别')
plt.ylabel('真实类别')
plt.title('轻量级 MLP 模型多分类混淆矩阵')
plt.tight_layout()
plt.savefig('实验图2_多分类混淆矩阵.png', dpi=300)

print("图表生成完毕！请在目录中查看。")

print("\n========== 4. 导出多分类 5 输出端侧量化模型 ==========")
converter_quant = tf.lite.TFLiteConverter.from_keras_model(mlp_model)
converter_quant.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_quant = converter_quant.convert()
with open('Multiclass_IDS_Quantized.tflite', 'wb') as f:
    f.write(tflite_quant)

print("✅ 终极 5 分类量化模型 (Multiclass_IDS_Quantized.tflite) 导出成功！")