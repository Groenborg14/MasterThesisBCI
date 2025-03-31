import mne
import os

subject = 13
runs = [1, 2, 3, 4, 5]

base_path = r"c:\Users\Christian\Desktop\Master Thesis\MI_data"

for run in runs:
    path = os.path.join(base_path, f"motorimagination_subject{subject}_run{run}.gdf")
    print(f"Trying to load: {path}")
    if not os.path.exists(path):
        print(f"❌ File does not exist: {path}")
        continue
    try:
        raw = mne.io.read_raw_gdf(path, preload=False)
        print(f"✅ Loaded successfully: run {run}")
    except Exception as e:
        print(f"❌ Error loading run {run}: {e}")
