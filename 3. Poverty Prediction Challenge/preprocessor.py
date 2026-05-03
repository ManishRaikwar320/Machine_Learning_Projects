from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import numpy as np

def DataPreprocessing(data):
    print(data.shape)
    # Identify missing value and fill with mode
    null_cols = []
    for col in data.columns:
        if data[col].isnull().sum() > 0:
            null_cols.append(col)
    
    for col in null_cols:
        data[col].fillna(data[col].mode()[0], inplace=True)

    data.drop(columns=['hhid', 'com', 'survey_id'], inplace=True)


    # One Hot Encoding for categorical columns
    ohe_cols = [
            'male', 'owner', 'water', 'toilet', 'sewer', 'elect', 'employed', 'any_nonagric', 'urban',
            'consumed100', 'consumed200', 'consumed300', 'consumed400',
        'consumed500', 'consumed600', 'consumed700', 'consumed800',
        'consumed900', 'consumed1000', 'consumed1100', 'consumed1200',
        'consumed1300', 'consumed1400', 'consumed1500', 'consumed1600',
        'consumed1700', 'consumed1800', 'consumed1900', 'consumed2000',
        'consumed2100', 'consumed2200', 'consumed2300', 'consumed2400',
        'consumed2500', 'consumed2600', 'consumed2700', 'consumed2800',
        'consumed2900', 'consumed3000', 'consumed3100', 'consumed3200',
        'consumed3300', 'consumed3400', 'consumed3500', 'consumed3600',
        'consumed3700', 'consumed3800', 'consumed3900', 'consumed4000',
        'consumed4100', 'consumed4200', 'consumed4300', 'consumed4400',
        'consumed4500', 'consumed4600', 'consumed4700', 'consumed4800',
        'consumed4900', 'consumed5000'
    ]


    for col in ohe_cols:
        ohe = OneHotEncoder(drop='first')
        data[col] = ohe.fit_transform(data[[col]]).toarray().astype(np.int64)


    # Label Encoding for categorical columns
    label_cols = ['water_source', 'sanitation_source', 'dweltyp', 'educ_max', 'sector1d']

    for col in label_cols:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col])


    return data