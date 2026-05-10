from pathlib import Path
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


PROJECT_ROOT = Path.cwd()
INPUT_DIR = PROJECT_ROOT / "results" / "binary_slow_healing_model"
OUTPUT_DIR = PROJECT_ROOT / "results" / "figures" / "binary_slow_healing"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

METRICS_PATH = INPUT_DIR / "binary_slow_healing_model_metrics.json"
IMPORTANCE_PATH = INPUT_DIR / "binary_slow_healing_feature_importance.csv"

with open(METRICS_PATH, "r", encoding="utf-8") as f:
    metrics = json.load(f)


# ============================================================
# Рисунок 1. Распределение классов
# ============================================================

class_dist = metrics["config"]["class_distribution"]
labels = list(class_dist.keys())
values = list(class_dist.values())

plt.figure(figsize=(5.5, 4))
plt.bar(labels, values)
plt.ylabel("Число наблюдений")
plt.xlabel("Класс")
plt.title("Распределение наблюдений по классам")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "binary_model_class_distribution.png", dpi=300)
plt.close()


# ============================================================
# Рисунок 2. Сравнение моделей по метрикам
# ============================================================

rows = []

dummy = metrics["dummy_metrics"]
rows.append({
    "model": "Dummy baseline",
    "accuracy": dummy["accuracy"],
    "balanced_accuracy": dummy["balanced_accuracy"],
    "f1_macro": dummy["f1_macro"],
})

for model_name, info in metrics["all_model_results"].items():
    m = info["metrics"]
    rows.append({
        "model": model_name,
        "accuracy": m["accuracy"],
        "balanced_accuracy": m["balanced_accuracy"],
        "f1_macro": m["f1_macro"],
    })

df_metrics = pd.DataFrame(rows)

name_map = {
    "Dummy baseline": "Dummy baseline",
    "logistic_regression": "Logistic Regression",
    "random_forest": "Random Forest",
    "extra_trees": "Extra Trees",
}
df_metrics["model_display"] = df_metrics["model"].map(name_map).fillna(df_metrics["model"])

x = np.arange(len(df_metrics))
width = 0.25

plt.figure(figsize=(10, 5))
plt.bar(x - width, df_metrics["accuracy"], width, label="Accuracy")
plt.bar(x, df_metrics["balanced_accuracy"], width, label="Balanced accuracy")
plt.bar(x + width, df_metrics["f1_macro"], width, label="Macro-F1")
plt.xticks(x, df_metrics["model_display"], rotation=20, ha="right")
plt.ylim(0, 1.05)
plt.ylabel("Значение метрики")
plt.xlabel("Модель")
plt.title("Сравнение моделей бинарной классификации")
plt.legend()
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "binary_model_metrics_comparison.png", dpi=300)
plt.close()


# ============================================================
# Рисунок 3. Матрица ошибок лучшей модели
# ============================================================

best = metrics["best_model_metrics"]
cm = np.array(best["confusion_matrix"])
classes = metrics["config"]["classes"]

plt.figure(figsize=(5.5, 5))
plt.imshow(cm)
plt.title("Матрица ошибок лучшей бинарной модели")
plt.xticks(np.arange(len(classes)), classes)
plt.yticks(np.arange(len(classes)), classes)
plt.xlabel("Предсказанный класс")
plt.ylabel("Истинный класс")

for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, str(cm[i, j]), ha="center", va="center", color="black")

plt.colorbar()
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "binary_model_confusion_matrix.png", dpi=300)
plt.close()


# ============================================================
# Рисунок 4. Важность признаков
# ============================================================

fi = pd.read_csv(IMPORTANCE_PATH)
top_n = 12
fi_top = fi.head(top_n).iloc[::-1]

plt.figure(figsize=(9, 6))
plt.barh(fi_top["feature"], fi_top["importance"])
plt.xlabel("Важность признака")
plt.title("Наиболее значимые признаки бинарной модели")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "binary_model_feature_importance.png", dpi=300)
plt.close()


print(f"Графики сохранены в: {OUTPUT_DIR}")
print("- binary_model_class_distribution.png")
print("- binary_model_metrics_comparison.png")
print("- binary_model_confusion_matrix.png")
print("- binary_model_feature_importance.png")
