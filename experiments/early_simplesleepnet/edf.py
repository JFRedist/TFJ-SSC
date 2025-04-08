import os
import numpy as np
import mne
import matplotlib.pyplot as plt

EDF_DIR = os.environ.get("SLEEP_EDF_DIR", "sleep-edf-database-expanded-1.0.0/sleep-cassette")

# 1. 读取仅含注释的 EDF 文件
annotations = mne.read_annotations(os.path.join(EDF_DIR, "SC4001EC-Hypnogram.edf"))


# 2. 查看注释基本信息
print("注释总数：", len(annotations))
print("前5个注释：")
for i in range(5):
    print(f"起始时间：{annotations.onset[i]:.1f}s，持续时间：{annotations.duration[i]:.1f}s，描述：{annotations.description[i]}")

# 3. 可视化注释时间轴（需结合信号数据）
# 如果有对应的信号数据文件，可以合并注释到原始数据中
# 假设信号数据文件为 "SC4001EC-EEG.edf"
try:
    from mne.io import read_raw_edf
    raw = read_raw_edf(os.path.join(EDF_DIR, "SC4001E0-PSG.edf"), preload=True)

    raw.set_annotations(annotations)  # 合并注释到信号数据
    fig = mne.viz.plot_events(
        events=mne.events_from_annotations(raw)[0],
        sfreq=raw.info["sfreq"],
        first_samp=raw.first_samp,
        event_id=mne.events_from_annotations(raw)[1]
    )
    fig.show()
except FileNotFoundError:
    print("未找到信号数据文件，仅显示注释信息。")

# 4. 自定义注释映射并生成阶段序列（如睡眠阶段）
stage_mapping = {
    "Sleep stage W": 0,
    "Sleep stage 1": 1,
    "Sleep stage 2": 2,
    "Sleep stage 3": 3,
    "Sleep stage 4": 3,
    "Sleep stage R": 4,
}

# 生成睡眠阶段时间序列（假设信号采样率为 100Hz）
sfreq = 100  # 需根据实际信号数据调整
n_samples = int(max(annotations.onset) + max(annotations.duration)) * sfreq
stage_sequence = np.full(n_samples, np.nan, dtype=object)

for onset, duration, desc in zip(annotations.onset, annotations.duration, annotations.description):
    start = int(onset * sfreq)
    end = int((onset + duration) * sfreq)
    stage_sequence[start:end] = stage_mapping.get(desc, np.nan)

stage_numeric = np.vectorize(stage_mapping.get)(stage_sequence)

# 绘制睡眠阶段序列
plt.figure(figsize=(12, 3))
plt.plot(stage_numeric, color="blue")
plt.yticks([0,1,2,3,4], ["Wake", "N1", "N2", "N3", "REM"])
plt.xlabel("Sample Index")
plt.title("Sleep Stage Annotation Sequence")
plt.show()