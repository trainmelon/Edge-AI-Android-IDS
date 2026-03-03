import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import os

print("========== 阶段 1：加载与清洗真实网络流量数据 ==========")
# NSL-KDD 数据集没有自带表头，我们需要手动为其指定列名
col_names = ["duration","protocol_type","service","flag","src_bytes",
    "dst_bytes","land","wrong_fragment","urgent","hot","num_failed_logins",
    "logged_in","num_compromised","root_shell","su_attempted","num_root",
    "num_file_creations","num_shells","num_access_files","num_outbound_cmds",
    "is_host_login","is_guest_login","count","srv_count","serror_rate",
    "srv_serror_rate","rerror_rate","srv_rerror_rate","same_srv_rate",
    "diff_srv_rate","srv_diff_host_rate","dst_host_count","dst_host_srv_count",
    "dst_host_same_srv_rate","dst_host_diff_srv_rate","dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate","dst_host_serror_rate","dst_host_srv_serror_rate",
    "dst_host_rerror_rate","dst_host_srv_rerror_rate","label", "difficulty_level"]

# 加载数据 (请确保 KDDTrain+.txt 在你的项目目录下)
# 如果文件没找到，请检查路径或文件名
try:
    df = pd.read_csv('KDDTrain+.txt', header=None, names=col_names)
    print(f"成功加载数据，总条数：{len(df)}")
except FileNotFoundError:
    print("错误：未找到 KDDTrain+.txt，请按第一步的指南下载并放到当前目录。")
    exit()

# 丢弃不需要的难度级别列
df.drop(['difficulty_level'], axis=1, inplace=True)

# 将具体的攻击类型（几十种）统一归类为“异常 (1)”，normal 归类为“正常 (0)”
df['label'] = df['label'].apply(lambda x: 0 if x == 'normal' else 1)

# 处理类别型特征 (例如 tcp, udp, http 等字符变为数字)
categorical_columns = ['protocol_type', 'service', 'flag']
for col in categorical_columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])

# 分离特征和标签
X = df.drop('label', axis=1).values
y = df['label'].values

# 特征归一化（极其重要：将所有数值缩放到相似范围，加快训练并提升精度）
scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"预处理完成！特征维度：{X_train.shape[1]} 维")

print("\n========== 阶段 2：构建并训练适用于移动端的轻量级模型 ==========")
# 运用深度学习对网络流量异常分析
model = tf.keras.Sequential([
    tf.keras.layers.Dense(32, activation='relu', input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(2, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 开始训练 (真实数据集较大，训练可能需要 1-2 分钟)
history = model.fit(X_train, y_train, epochs=10, batch_size=256, validation_data=(X_test, y_test))

print("\n========== 阶段 3：量化压缩并导出为 Android TFLite 格式 ==========")
# 实例化 TFLite 转换器
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# 开启默认优化，自动执行权重量化 (将 float32 压缩为 int8)
# 大幅压缩模型体积和计算量
converter.optimizations = [tf.lite.Optimize.DEFAULT]

tflite_quant_model = converter.convert()

tflite_filename = 'RealData_IDS_Quantized.tflite'
with open(tflite_filename, 'wb') as f:
    f.write(tflite_quant_model)

print(f"\n✅ 大功告成！基于真实数据集的量化模型已生成：{os.path.abspath(tflite_filename)}")