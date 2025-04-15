from datasets import load_dataset

dataset = load_dataset("json", data_files="data/processed_financial_data.json")["train"]
print(dataset[0])