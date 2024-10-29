import pandas as pd
import json


def load_csv(csv_path):
    return pd.read_csv(csv_path)


def write_json(data, output_path):
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=4)
