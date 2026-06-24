import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import joblib # for saving the label encoder
import os  # for checking file existence


# Making the folders for model and data if they don't exist
os.makedirs('model',exist_ok = True) 
os.makedirs('data',exist_ok = True)



# Function to convert diagnosis codes to their respective groups
def map_icd9_to_group(code):
    try:
        code = str(code).strip() # Ensure code is a string and remove any leading/trailing whitespace
        if code.startswith('E') or code.startswith('V'):
            return 'Other'
        c = float(code)
        if 390 <= c <= 459 or c == 785: 
            return 'Circulatory'
        if 460 <= c <= 519 or c == 786: 
            return 'Respiratory'
        if 520 <= c <= 579 or c == 787: 
            return 'Digestive'
        if 250 <= c <= 250.99:          
            return 'Diabetes'
        if 800 <= c <= 999:             
            return 'Injury'
        if 710 <= c <= 739:             
            return 'Musculoskeletal'
        if 580 <= c <= 629 or c == 788: 
            return 'Genitourinary'
        if 140 <= c <= 239:             
            return 'Neoplasms'
        return 'Other'
    except:
        return 'Other'
    


def preprocess(input_path='data/diabetic_data.csv', output_path = 'data/preprocessed_data.csv'):
    print("Loading datasets...")

    # Load the dataset
    df = pd.read_csv(input_path)

    print(df.head(20))

    print(f"Shape : {df.shape}")

    # Replacing the ? with NaN
    df.replace('?', np.nan, inplace = True)


    # dropping the unnecessary columns
    drop_columns = ['weight', 'payer_code', 'medical_specialty','encounter_id', 'patient_nbr']
    df.drop(columns= [c for c in drop_columns if c in df.columns] , inplace = True)
    print(f"Shape after dropping columns : {df.shape}")

    # Dropping the columns where race is missing 
    df.dropna(subset=["race"] , inplace = True)

    # Binarize the target variable readmitted
    df['readmitted_30'] = (df['readmitted'] == '<30').astype(int)
    df.drop(columns=['readmitted'] , inplace = True)
    print(f"Readmission rate (<30 days) : {df['readmitted_30'].mean():.2%}")

    # Mapping the diagnosis codes to their respective groups
    for col in ['diag_1', 'diag_2', 'diag_3']:
        if col in df.columns:
            df[col] = df[col].apply(map_icd9_to_group)

    # Mapping the age column to the mid-point of the age range
    age_map = {
        '[0-10)' : 5,'[10-20)': 15, '[20-30)': 25, '[30-40)': 35,
        '[40-50)': 45, '[50-60)': 55, '[60-70)': 65, '[70-80)': 75,
        '[80-90)': 85, '[90-100)': 95
    }
    df['age'] = df['age'].map(age_map)

    # Encode medication change columns (Ch/No -> 1/0)
    change_cols = ['change', 'diabetesMed']
    for col in change_cols:
        if col in df.columns:
            df[col] = (df[col] == 'Ch').astype(int) if col == 'change' else (df[col] == 'Yes').astype(int)

    # Simplify medication columns: Down/Steady/Up/No -> 0/1/2/-1
    med_map = {'No': 0, 'Steady': 1, 'Up': 2, 'Down': -1}
    med_cols = ['metformin','repaglinide','nateglinide','chlorpropamide',
                'glimepiride','acetohexamide','glipizide','glyburide',
                'tolbutamide','pioglitazone','rosiglitazone','acarbose',
                'miglitol','troglitazone','tolazamide','examide',
                'citoglipton','insulin','glyburide-metformin',
                'glipizide-metformin','glimepiride-pioglitazone',
                'metformin-rosiglitazone','metformin-pioglitazone']
    for col in med_cols:
        if col in df.columns:
            df[col] = df[col].map(med_map).fillna(0).astype(int)

    # Label encode remaining categoricals
    cat_cols = df.select_dtypes(include='object').columns.tolist()
    encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = df[col].fillna('Unknown')
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le
 
    joblib.dump(encoders, 'model/encoders.pkl')
    print(f"  Saved encoders for {len(encoders)} columns")
 
    # Fill any remaining numeric NaN with median
    df.fillna(df.median(numeric_only=True), inplace=True)
 
    df.to_csv(output_path, index=False)
    print(f"  Saved processed data -> {output_path}")
    print(f"  Final shape: {df.shape}")
    print(f"  Columns: {list(df.columns)}")
    return df
 
if __name__ == '__main__':
    preprocess()