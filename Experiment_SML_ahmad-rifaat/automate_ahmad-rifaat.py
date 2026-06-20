import os
import sys
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder

def preprocess_data(input_path, output_dir):
    if not os.path.exists(input_path):
        print(f"Error: File {input_path} tidak ditemukan!")
        sys.exit(1)
        
    print("Memulai proses otomatisasi preprocessing data...")
    df = pd.read_csv(input_path)
    
    # hapus duplikat
    df = df.drop_duplicates()
    
    # menangani missing value
    df['income'] = df['income'].fillna(df['income'].median())
    df['credit_score'] = df['credit_score'].fillna(df['credit_score'].mean())
    
    # encoding data
    le = LabelEncoder()
    df['employment_type'] = le.fit_transform(df['employment_type'])
    
    # normalisasi
    scaler = StandardScaler()
    features_to_scale = ['age', 'income', 'loan_amount', 'credit_score']
    df[features_to_scale] = scaler.fit_transform(df[features_to_scale])
    
    # Drop kolom tidak penting untuk training
    df = df.drop(columns=['customer_id'])
    
    # directory
    os.makedirs(output_dir, exist_ok=True)
    processed_path = os.path.join(output_dir, 'credit_data_processed.csv')
    
    # save
    df.to_csv(processed_path, index=False)
    print(f"Otomatisasi sukses! Data siap latih disimpan di: {processed_path}")

if __name__ == "__main__":
    # Menentukan path input/output
    RAW_DATA_PATH = os.path.join('namadataset_raw', 'credit_data.csv')
    OUTPUT_DIR = 'namadataset_preprocessing'
    
    preprocess_data(RAW_DATA_PATH, OUTPUT_DIR)