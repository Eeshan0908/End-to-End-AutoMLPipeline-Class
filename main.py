import os
import pandas as pd

from src.automl_pipeline import AutoMLPipeline


data_folder = "data"

csv_files = [file for file in os.listdir(data_folder)if file.endswith(".csv")]

if len(csv_files) == 0:

    raise FileNotFoundError("No CSV file found in data folder.")

csv_path = os.path.join(data_folder,csv_files[0])

print(f"\nDataset Loaded: {csv_files[0]}")

df = pd.read_csv(csv_path)

print("\nAvailable Columns:")

print(df.columns.tolist())

target_column = input("\nEnter target column: ").strip()

if target_column not in df.columns:

    raise ValueError(
        f"{target_column} not found.\n"
        f"Available columns:\n"
        f"{list(df.columns)}"
    )

pipeline = AutoMLPipeline(df=df,target_column=target_column)

result = pipeline.run()


# Convert result to DataFrame

if isinstance(result, dict):

    if "results" in result:

        results_df = pd.DataFrame(result["results"])
    
    else:
        
        results_df = pd.DataFrame([result])

else:

    results_df = pd.DataFrame(result)


# Sort Results


metric = None

for col in ["accuracy","r2_score","score"]:

    if col in results_df.columns:

        metric = col
        break

if metric:

    results_df = (results_df.sort_values(by=metric,ascending=False).reset_index(drop=True))


# Save Results


os.makedirs("outputs",exist_ok=True)

results_df.to_csv("outputs/model_results.csv",index=False)


# Display Results


print("\nModel Comparison")

print(results_df.to_string(index=False))


# Best Model


if len(results_df) > 0:

    print("\nBest Model")

    print(results_df.iloc[0])

print("\nResults Saved:")

print("outputs/model_results.csv")