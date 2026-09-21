import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load datasets
df1 = pd.read_csv('dataset/Tuesday-WorkingHours.pcap_ISCX.csv', low_memory=True)
df2 = pd.read_csv('dataset/Wednesday-workingHours.pcap_ISCX.csv', low_memory=True)
df3 = pd.read_csv('dataset/Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv', low_memory=True)

dataset = pd.concat([df1, df2, df3], ignore_index=True)
dataset.columns = dataset.columns.str.strip()

# Features and target
X = dataset.iloc[:, :-1]
y = dataset.iloc[:, -1]

X = X.apply(pd.to_numeric, errors='coerce')
X.replace([np.inf, -np.inf], np.nan, inplace=True)

# Imports
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.kernel_approximation import Nystroem
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# Encode target
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)

print("Classes:", labelencoder_y.classes_)

# Output folders
os.makedirs("Results/Screenshots", exist_ok=True)
os.makedirs("Results/Confusion_Matrices", exist_ok=True)

# Experiment parameters
kpca_components_list = [5, 10, 15]
test_sizes = [0.2, 0.3, 0.4, 0.5, 0.6]

algorithms = [
    "Decision Tree",
    "Random Forest",
    "Gradient Boosting",
    "XGBoost",
    "LightGBM"
]

# Kernel PCA parameters
kernel = "rbf"
gamma = 15

results = []

total_experiments = (
    len(kpca_components_list)
    * len(test_sizes)
    * len(algorithms)
)

experiment_number = 0

# Run experiments
for n_components in kpca_components_list:

    for test_size in test_sizes:

        print(f"\n{'=' * 70}")
        print(
            f"Kernel PCA Components: {n_components} | "
            f"Test Size: {test_size}"
        )
        print(f"Kernel: {kernel} | Gamma: {gamma}")
        print(f"{'=' * 70}")

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=0,
            stratify=y
        )

        # Imputation
        imputer = SimpleImputer(strategy='mean')

        X_train = imputer.fit_transform(X_train)
        X_test = imputer.transform(X_test)

        # Standardization
        scaler = StandardScaler()

        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # Approximate RBF Kernel PCA without creating an n_samples x n_samples matrix
        kpca = Nystroem(
            n_components=n_components,
            kernel=kernel,
            gamma=gamma,
            random_state=0
        )

        X_train_kpca = kpca.fit_transform(X_train)
        X_test_kpca = kpca.transform(X_test)

        # Run algorithms
        for algorithm_name in algorithms:

            experiment_number += 1

            print(
                f"\n[{experiment_number}/{total_experiments}] "
                f"{algorithm_name} | "
                f"KPCA={n_components} | "
                f"Test={test_size}"
            )

            # Select algorithm
            if algorithm_name == "Decision Tree":

                model = DecisionTreeClassifier(
                    criterion='entropy',
                    random_state=0
                )

            elif algorithm_name == "Random Forest":

                model = RandomForestClassifier(
                    n_estimators=200,
                    criterion='entropy',
                    random_state=0
                )

            elif algorithm_name == "Gradient Boosting":

                model = GradientBoostingClassifier()

            elif algorithm_name == "XGBoost":

                model = XGBClassifier()

            elif algorithm_name == "LightGBM":

                model = LGBMClassifier()

            # Train and predict
            model.fit(
                X_train_kpca,
                y_train
            )

            y_pred = model.predict(
                X_test_kpca
            )

            # Metrics
            mae = mean_absolute_error(
                y_test,
                y_pred
            )

            mse = mean_squared_error(
                y_test,
                y_pred
            )

            r2 = r2_score(
                y_test,
                y_pred
            )

            accuracy = accuracy_score(
                y_test,
                y_pred
            )

            precision = precision_score(
                y_test,
                y_pred,
                average='weighted',
                zero_division=0
            )

            recall = recall_score(
                y_test,
                y_pred,
                average='weighted',
                zero_division=0
            )

            f1 = f1_score(
                y_test,
                y_pred,
                average='weighted',
                zero_division=0
            )

            confusion = confusion_matrix(
                y_test,
                y_pred,
                labels=np.arange(len(labelencoder_y.classes_))
            )

            print(
                f"Precision={precision:.4f} | "
                f"Recall={recall:.4f} | "
                f"F1 Score={f1:.4f}"
            )
            print("Confusion Matrix:")
            print(confusion)

            confusion_filename = (
                f"{algorithm_name.replace(' ', '_')}_"
                f"KPCA_{n_components}_"
                f"Gamma_{gamma}_"
                f"Test_{str(test_size).replace('.', '_')}.csv"
            )

            pd.DataFrame(
                confusion,
                index=labelencoder_y.classes_,
                columns=labelencoder_y.classes_
            ).to_csv(
                os.path.join(
                    "Results",
                    "Confusion_Matrices",
                    confusion_filename
                )
            )

            print(
                f"MAE={mae:.4f} | "
                f"MSE={mse:.4f} | "
                f"R2={r2:.4f} | "
                f"Accuracy={accuracy:.4f}"
            )

            # Store results
            results.append({
                "Algorithm": algorithm_name,
                "KPCA Components": n_components,
                "Test Size": test_size,
                "Kernel": kernel,
                "Gamma": gamma,
                "Train Size": len(y_train),
                "Test Samples": len(y_test),
                "Accuracy": accuracy,
                "Precision": precision,
                "Recall": recall,
                "F1 Score": f1
            })

            # Save CSV after every experiment
            pd.DataFrame(results).to_csv(
                "Results/all_kpca_experiment_results.csv",
                index=False
            )

            # Screenshot
            figure, (metrics_axis, matrix_axis) = plt.subplots(
                1,
                2,
                figsize=(18, 8),
                gridspec_kw={"width_ratios": [1, 1.4]}
            )

            metrics_axis.axis("off")

            metrics_axis.text(
                0.5,
                0.94,
                "Kernel PCA Machine Learning Experiment",
                ha="center",
                fontsize=18,
                fontweight="bold"
            )

            metrics_axis.text(
                0.5,
                0.87,
                algorithm_name,
                ha="center",
                fontsize=16
            )

            result_text = (
                f"KPCA Components : {n_components}\n"
                f"Kernel          : {kernel}\n"
                f"Gamma           : {gamma}\n"
                f"Test Size       : {test_size}\n"
                f"Train Samples   : {len(y_train)}\n"
                f"Test Samples    : {len(y_test)}\n\n"
                f"Accuracy            : {accuracy:.4f}\n"
                f"Precision           : {precision:.4f}\n"
                f"Recall              : {recall:.4f}\n"
                f"F1 Score            : {f1:.4f}"
            )

            metrics_axis.text(
                0.5,
                0.50,
                result_text,
                ha="center",
                va="center",
                fontsize=12,
                family="monospace"
            )

            matrix_image = matrix_axis.imshow(
                confusion,
                interpolation="nearest",
                cmap="Blues"
            )
            matrix_axis.set_title("Confusion Matrix", fontsize=16)
            matrix_axis.set_xlabel("Predicted label")
            matrix_axis.set_ylabel("True label")
            matrix_axis.set_xticks(np.arange(len(labelencoder_y.classes_)))
            matrix_axis.set_yticks(np.arange(len(labelencoder_y.classes_)))
            matrix_axis.set_xticklabels(
                labelencoder_y.classes_,
                rotation=90,
                fontsize=7
            )
            matrix_axis.set_yticklabels(
                labelencoder_y.classes_,
                fontsize=7
            )
            figure.colorbar(matrix_image, ax=matrix_axis, fraction=0.046)

            for row_index in range(confusion.shape[0]):
                for column_index in range(confusion.shape[1]):
                    matrix_axis.text(
                        column_index,
                        row_index,
                        confusion[row_index, column_index],
                        ha="center",
                        va="center",
                        fontsize=6
                    )

            metrics_axis.text(
                0.5,
                0.05,
                f"Experiment "
                f"{experiment_number}/"
                f"{total_experiments}",
                ha="center",
                fontsize=11
            )

            filename = (
                f"{algorithm_name.replace(' ', '_')}_"
                f"KPCA_{n_components}_"
                f"Gamma_{gamma}_"
                f"Test_{str(test_size).replace('.', '_')}.png"
            )

            figure.tight_layout()
            figure.savefig(
                os.path.join(
                    "Results",
                    "Screenshots",
                    filename
                ),
                dpi=150,
                bbox_inches="tight"
            )

            plt.close()

# Final CSV
results_df = pd.DataFrame(results)

results_df.to_csv(
    "Results/all_kpca_experiment_results.csv",
    index=False
)

# Best result for each algorithm
best_results = (
    results_df
    .sort_values(
        "Accuracy",
        ascending=False
    )
    .groupby("Algorithm")
    .first()
    .reset_index()
)

best_results.to_csv(
    "Results/best_kpca_results_by_algorithm.csv",
    index=False
)

print("\n" + "=" * 70)
print("ALL KERNEL PCA EXPERIMENTS COMPLETED")
print("=" * 70)
print(f"Experiments completed: {len(results_df)}")
print(
    "All results: "
    "Results/all_kpca_experiment_results.csv"
)
print(
    "Best results: "
    "Results/best_kpca_results_by_algorithm.csv"
)
print("Screenshots: Results/Screenshots/")