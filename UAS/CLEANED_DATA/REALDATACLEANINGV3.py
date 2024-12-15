import pandas as pd

# Membaca file CSV dengan tanda titik koma (;) sebagai pemisah
df = pd.read_csv('Shopping_Mall_Customer_Segmentation _Data.csv', sep=';')

# 1. Membersihkan kolom Annual Income dengan membaginya dengan 1000 dan membulatkan
df['Annual Income'] = (df['Annual Income'] / 1000).round(0).astype(int)

# 3. Membagi menjadi 3 grup berdasarkan age (tambahkan ini sebelum membagi berdasarkan gender)
def categorize_age(age):
    if age < 30:
        return 'young'
    elif 30 <= age < 60:
        return 'adult'
    else:
        return 'old'

df['Age Group'] = df['Age'].apply(categorize_age)

# 2. Memisahkan berdasarkan Gender (lakukan ini setelah menambahkan kolom 'Age Group')
df_male = df[df['Gender'] == 'Male']
df_female = df[df['Gender'] == 'Female']

# Menyimpan file 2 dan 3 (berdasarkan gender)
df_male.to_csv('cleaned_male.csv', index=False)
df_female.to_csv('cleaned_female.csv', index=False)

# Menyimpan file 4, 5, dan 6 (berdasarkan grup umur)
df_young = df[df['Age Group'] == 'young']
df_adult = df[df['Age Group'] == 'adult']
df_old = df[df['Age Group'] == 'old']

df_young.to_csv('cleaned_young.csv', index=False)
df_adult.to_csv('cleaned_adult.csv', index=False)
df_old.to_csv('cleaned_old.csv', index=False)

# 4. Menggabungkan Point 1 dan Point 2 (Gender dan Age Group)
# Menggabungkan male dan female untuk setiap grup umur
df_male_young = df_male[df_male['Age Group'] == 'young']
df_male_adult = df_male[df_male['Age Group'] == 'adult']
df_male_old = df_male[df_male['Age Group'] == 'old']

df_female_young = df_female[df_female['Age Group'] == 'young']
df_female_adult = df_female[df_female['Age Group'] == 'adult']
df_female_old = df_female[df_female['Age Group'] == 'old']

# Menyimpan file 7 sampai 12
df_male_young.to_csv('male_young.csv', index=False)
df_male_adult.to_csv('male_adult.csv', index=False)
df_male_old.to_csv('male_old.csv', index=False)

df_female_young.to_csv('female_young.csv', index=False)
df_female_adult.to_csv('female_adult.csv', index=False)
df_female_old.to_csv('female_old.csv', index=False)

# Daftar file yang disimpan
files_created = [
    'cleaned_annual_income.csv', 
    'cleaned_male.csv', 
    'cleaned_female.csv', 
    'cleaned_young.csv', 
    'cleaned_adult.csv', 
    'cleaned_old.csv',
    'male_young.csv', 
    'male_adult.csv', 
    'male_old.csv', 
    'female_young.csv', 
    'female_adult.csv', 
    'female_old.csv'
]

files_created
