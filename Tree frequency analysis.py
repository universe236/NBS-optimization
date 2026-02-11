import pandas as pd
import os

def analyze_tree_species(file_path):
    output_dir = os.path.dirname(file_path)
    df = pd.read_csv(file_path)

    if len(df.columns) <= 3:
        raise ValueError("DataFrame has fewer than 4 columns, cannot access index 3.")

    species_column = df.columns[3]
    original_count = len(df)

    df[species_column] = df[species_column].fillna('MISSING_VALUE')

    mask_species = df[species_column].str.contains('SPECIES', case=False)
    mask_unknown = df[species_column] == 'UNKNOWN UNKNOWN - UNKNOWN'
    mask_missing = df[species_column] == 'MISSING_VALUE'

    mask_remove = mask_species | mask_unknown | mask_missing
    df_filtered = df[~mask_remove]

    species_counts = df_filtered[species_column].value_counts()
    species_percentages = (species_counts / len(df_filtered) * 100).round(2)

    results = pd.DataFrame({
        'Count': species_counts,
        'Percentage': species_percentages
    })

    print(f"\nOriginal record count: {original_count}")
    print(f"Valid record count: {len(df_filtered)}")
    print(f"Removed record count: {original_count - len(df_filtered)}")
    print("\nTree species analysis results:")
    print(results.to_string())

    output_file = os.path.join(output_dir, 'tree_species_analysis.csv')
    results.to_csv(output_file)
    print(f"\nResults saved to: {output_file}")

    print(f"\nDetails of removed records:")
    print(f"Records containing 'SPECIES': {mask_species.sum()}")
    print(f"Records with 'UNKNOWN UNKNOWN - UNKNOWN': {mask_unknown.sum()}")
    print(f"Missing/NaN records: {mask_missing.sum()}")

file_path = r"/path/to/your/file/tree_inventory_2024.csv"
try:
    analyze_tree_species(file_path)
except Exception as e:
    print(f"An error occurred: {str(e)}")
