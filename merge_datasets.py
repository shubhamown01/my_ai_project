import pandas as pd
import glob
import os


def merge_all_csv():
    """
    raw_data_folder ke andar saari CSV files ko ek master dataset mein merge karna.

    Features:
    - Duplicate display_name hata deta hai
    - Null values ko 'Unknown' se fill karta hai
    - Column names clean karta hai (leading/trailing spaces hata deta hai)
    - Master file khud mein merge nahi hoti
    """
    input_folder = 'raw_data_folder'
    output_file  = os.path.join(input_folder, 'global_object_dataset.csv')

    if not os.path.exists(input_folder):
        print(f"[ERROR] '{input_folder}' folder nahi mila! Pehle banayein.")
        return

    # Saari CSV files dhoondhna (master ko exclude karo)
    all_files = [
        f for f in glob.glob(os.path.join(input_folder, "*.csv"))
        if 'global_object_dataset.csv' not in f
    ]

    if not all_files:
        print("[WARNING] Koi nayi CSV file nahi mili merge karne ke liye.")
        return

    print(f"[INFO] {len(all_files)} file(s) merge hone ke liye taiyar hain...")

    loaded_dfs = []
    skipped    = 0

    for filepath in all_files:
        try:
            df = pd.read_csv(filepath, index_col=None, header=0, low_memory=False)
            # Column names clean karo
            df.columns = df.columns.str.strip()
            loaded_dfs.append(df)
            print(f"  [OK] {os.path.basename(filepath)}  ({len(df):,} rows)")
        except Exception as e:
            print(f"  [SKIP] {os.path.basename(filepath)} — {e}")
            skipped += 1

    if not loaded_dfs:
        print("[ERROR] Koi bhi file load nahi ho paya.")
        return

    # Merge karo
    master = pd.concat(loaded_dfs, axis=0, ignore_index=True)
    before = len(master)

    # Duplicates hataao (display_name basis par, pehla rakho)
    if 'display_name' in master.columns:
        master.drop_duplicates(subset=['display_name'], keep='first', inplace=True)
    else:
        master.drop_duplicates(inplace=True)

    after = len(master)

    # Null values fill karo
    master.fillna("Unknown", inplace=True)

    # Save
    master.to_csv(output_file, index=False, encoding='utf-8')

    file_size_kb = os.path.getsize(output_file) / 1024

    print()
    print("=" * 45)
    print("  MASTER DATASET READY!")
    print("=" * 45)
    print(f"  Files merged       : {len(loaded_dfs)}")
    print(f"  Files skipped      : {skipped}")
    print(f"  Total records      : {after:,}")
    print(f"  Duplicates removed : {before - after:,}")
    print(f"  File size          : {file_size_kb:.1f} KB")
    print(f"  Saved to           : {output_file}")
    print("=" * 45)

    if 'yolo_name' in master.columns:
        print("\nYOLO class distribution (top 15):")
        print(master['yolo_name'].value_counts().head(15).to_string())

    if 'confirmed' in master.columns:
        confirmed_count = (master['confirmed'] == True).sum()
        print(f"\nConfirmed records : {confirmed_count:,} / {after:,}")


if __name__ == "__main__":
    merge_all_csv()