# ============================================================
# EMNIST Letters Classification using HOG + SVM + LOOCV (Optimized Parallel)
# Author: Nur Shakinah
# ============================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import LeaveOneOut
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, f1_score
from skimage.feature import hog
from tqdm import tqdm
import seaborn as sns
import joblib
import random
from joblib import Parallel, delayed
from sklearn.base import clone

# ============================================================
# 1️⃣ Persiapan dan Setup Awal
# ============================================================
np.random.seed(42)
random.seed(42)

# ============================================================
# 2️⃣ Load Dataset CSV
# ============================================================
print("🔹 Membaca dataset...")
data_path = r"C:\Users\user\Documents\Kuliah Mekatronika\SEMESTER 5 MK\MACHINE VISION\Data\emnist-letters-train.csv"
df = pd.read_csv(data_path)

# Label: kolom pertama, Data: kolom selanjutnya
labels = df.iloc[:, 0].values
images = df.iloc[:, 1:].values / 255.0  # normalisasi ke 0-1
print("✅ Dataset terbaca:", df.shape)

# ============================================================
# 3️⃣ Sampling data: total 13.000 (26 kelas x 500)
# ============================================================
sample_per_class = 500
num_classes = 26
sampled_images = []
sampled_labels = []

print(" Melakukan sampling seimbang...")
for cls in range(1, num_classes + 1):
    cls_indices = np.where(labels == cls)[0]
    selected = np.random.choice(cls_indices, sample_per_class, replace=False)
    sampled_images.append(images[selected])
    sampled_labels.append(labels[selected])

X = np.vstack(sampled_images)
y = np.hstack(sampled_labels)
print("✅ Selesai sampling:", X.shape, "data")

# ============================================================
# 4️⃣ Ekstraksi fitur HOG (disimpan agar tidak perlu diulang)
# ============================================================
hog_path = r"C:\Users\user\Documents\Kuliah Mekatronika\SEMESTER 5 MK\MACHINE VISION\UTS_emnist-hog-svm\result\hog_features.npy"
label_path = r"C:\Users\user\Documents\Kuliah Mekatronika\SEMESTER 5 MK\MACHINE VISION\UTS_emnist-hog-svm\result\hog_labels.npy"

if os.path.exists(hog_path) and os.path.exists(label_path):
    print("📂 Memuat fitur HOG dari file...")
    hog_features = np.load(hog_path)
    y = np.load(label_path)
else:
    print("⚙️ Mengekstraksi fitur HOG...")
    hog_features = []
    for img in tqdm(X, desc="Extracting HOG"):
        img_reshaped = img.reshape((28, 28))
        features = hog(
            img_reshaped,
            orientations=9,
            pixels_per_cell=(8, 8),
            cells_per_block=(2, 2),
            block_norm='L2-Hys'
        )
        hog_features.append(features)
    hog_features = np.array(hog_features)
    np.save(hog_path, hog_features)
    np.save(label_path, y)
    print("💾 Fitur HOG disimpan untuk digunakan kembali.")

print("✅ Ekstraksi fitur selesai:", hog_features.shape)

# ============================================================
# 5️⃣ Standarisasi fitur
# ============================================================
scaler = StandardScaler()
hog_scaled = scaler.fit_transform(hog_features)

# ============================================================
# 6️⃣ Klasifikasi dengan SVM (RBF kernel)
# ============================================================
model_template = svm.SVC(kernel='rbf', C=10, gamma='scale')

# ============================================================
# 7️⃣ Evaluasi dengan Leave-One-Out Cross Validation (PARALEL)
# ============================================================
subset_size = 13000  # ubah sesuai kemampuan CPU
indices = np.random.choice(len(hog_scaled), subset_size, replace=False)
X_subset = hog_scaled[indices]
y_subset = y[indices]

loo = LeaveOneOut()

print(f"🧮 Melakukan LOOCV paralel pada {subset_size} sampel...")

def train_and_predict(train_idx, test_idx):
    model = clone(model_template)
    model.fit(X_subset[train_idx], y_subset[train_idx])
    y_true_i = y_subset[test_idx][0]
    y_pred_i = model.predict(X_subset[test_idx])[0]
    return y_true_i, y_pred_i

results = Parallel(n_jobs=6, backend='loky')(
    delayed(train_and_predict)(train_idx, test_idx)
    for train_idx, test_idx in tqdm(loo.split(X_subset), total=subset_size)
)

y_true, y_pred = zip(*results)

# ============================================================
# 8️⃣ Evaluasi Metrik
# ============================================================
cm = confusion_matrix(y_true, y_pred, labels=range(1, num_classes + 1))
acc = accuracy_score(y_true, y_pred)
prec = precision_score(y_true, y_pred, average='macro', zero_division=0)
f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)

print("\n📊 HASIL EVALUASI:")
print("Accuracy :", acc)
print("Precision:", prec)
print("F1-Score :", f1)

# ============================================================
# 9️⃣ Simpan hasil ke file
# ============================================================
os.makedirs(r"C:\Users\user\Documents\Kuliah Mekatronika\SEMESTER 5 MK\MACHINE VISION\UTS_emnist-hog-svm\result", exist_ok=True)
np.savetxt(r"C:\Users\user\Documents\Kuliah Mekatronika\SEMESTER 5 MK\MACHINE VISION\UTS_emnist-hog-svm\result\confusion_matrix.csv", cm, fmt='%d', delimiter=',')
with open(r"C:\Users\user\Documents\Kuliah Mekatronika\SEMESTER 5 MK\MACHINE VISION\UTS_emnist-hog-svm\result\evaluation_report.txt", "w") as f:
    f.write("Accuracy: {:.4f}\n".format(acc))
    f.write("Precision: {:.4f}\n".format(prec))
    f.write("F1-score: {:.4f}\n".format(f1))

# ============================================================
# 🔟 Visualisasi Confusion Matrix
# ============================================================
plt.figure(figsize=(10, 8))
sns.heatmap(cm, cmap="Blues", cbar=True, xticklabels=range(1, 27), yticklabels=range(1, 27))
plt.title("Confusion Matrix (LOOCV Parallel)")
plt.xlabel("Predicted")
plt.ylabel("True Label")
plt.tight_layout()
plt.savefig(r"C:\Users\user\Documents\Kuliah Mekatronika\SEMESTER 5 MK\MACHINE VISION\UTS_emnist-hog-svm\result\confusion_matrix.png")
plt.close()
print("📁 Semua hasil disimpan di folder result/")

# ============================================================
# 11️⃣ Visualisasi Contoh Prediksi Gambar
# ============================================================
print("🖼️ Menampilkan contoh hasil prediksi...")
num_show = 10
indices_show = np.random.choice(range(subset_size), num_show, replace=False)

plt.figure(figsize=(12, 4))
for i, idx in enumerate(indices_show):
    img = X[indices[idx]].reshape(28, 28)
    true_label = y_subset[idx]
    pred_label = model_template.fit(X_subset, y_subset).predict(X_subset[idx].reshape(1, -1))[0]

    plt.subplot(2, 5, i + 1)
    plt.imshow(img, cmap='gray')
    plt.title(f"T:{chr(true_label + 64)} / P:{chr(pred_label + 64)}")
    plt.axis('off')

plt.suptitle("Contoh Hasil Prediksi EMNIST Letters (T=True, P=Predicted)")
plt.tight_layout()
plt.savefig(r"C:\Users\user\Documents\Kuliah Mekatronika\SEMESTER 5 MK\MACHINE VISION\UTS_emnist-hog-svm\result\sample_predictions.png")
plt.show()
