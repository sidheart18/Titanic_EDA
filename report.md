# 🚢 **Titanic Dataset — Exploratory Data Analysis Report**

---

## **1. Introduction**

This report presents the Exploratory Data Analysis (EDA) performed on the Titanic dataset.
The goal was to understand the data, handle missing values, create useful new features, and generate visual insights using Python libraries such as **Pandas, NumPy, Matplotlib, and Seaborn**.

---

## **2. Loading the Dataset**

The dataset was loaded from the given file path:

```python
df = pd.read_csv(r'E:\UPGRAD\python\PROJECT_PY\titanic_project\datasets\train.csv')
```

Initial functions used:

* `df.head()` → Previewed the first 5 rows
* `df.info()` → Checked data types and missing values
* `df.describe()` → Observed statistical summary
* `df.isnull().sum()` → Counted missing values in each column

### **Key Observations Before Cleaning:**

* **Age** column had missing values
* **Embarked** column had 2 missing entries
* **Cabin** column had many missing values (majority of rows)
* Dataset contained 891 records and 12 columns

---

## **3. Data Cleaning Steps**

The following cleaning was performed step by step:

### **3.1 Missing Age Values**

```python
df["Age"] = df["Age"].fillna(df['Age'].median())
```

* Median age was used because Age is continuous and slightly skewed.

### **3.2 Missing Embarked Values**

```python
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
```

* Mode is suitable for categorical values.

### **3.3 Dropping Cabin Column**

```python
df.drop(columns=['Cabin'], inplace=True)
```

* Cabin had too many missing values to be useful.

---

## **4. Feature Engineering**

Three new features were created to discover additional patterns:

### **4.1 FamilySize**

```python
df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
```

* Counts total family members aboard (self + siblings/spouse + parents/children)

### **4.2 IsAlone**

```python
df['IsAlone'] = (df['FamilySize'] == 1).astype(int)
```

* 1 → Passenger is alone
* 0 → Traveling with family

### **4.3 Title Extraction**

```python
df['Title'] = df['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
```

* Extracted titles like *Mr, Mrs, Miss, Master* from the Name column.

---

## **5. Visual Exploration (Using Seaborn & Matplotlib)**

Below are visual insights directly based on the plots you generated.

---

### **5.1 Survival Count Plot**

```python
sns.countplot(data=df, x='Survived')
plt.title("Survivors Count")
```

#### **Observations:**

* The number of passengers who **did not survive** is significantly higher.
* Dataset is imbalanced: approx **62% did not survive**, **38% survived**.

---

### **5.2 Sex vs Survival**

```python
sns.countplot(data=df, x='Sex', hue='Survived')
plt.title("Sex vs Survival")
```

#### **Observations:**

* **Females survived much more than males.**
* Most males did **not** survive.
* Gender shows a strong relationship with survival.

---

### **5.3 Age Distribution**

```python
plt.hist(df['Age'], bins=30)
plt.title("Age Distribution")
```

#### **Observations:**

* Majority of passengers were between **20 and 40 years old**.
* There were also children on board (Age < 10).
* The distribution is slightly right-skewed.

---

### **5.4 Passenger Class vs Survival**

```python
sns.countplot(data=df, x='Pclass', hue='Survived')
plt.title("Class vs Survival")
```

#### **Observations:**

* **1st class passengers** had the highest survival rate.
* **3rd class passengers** had the lowest survival rate.
* Pclass plays an important role in survival.

---

### **5.5 Correlation Heatmap**

```python
numeric_df = df.select_dtypes(include=['int64', 'float64'])
sns.heatmap(numeric_df.corr(), annot=True, fmt=".2f")
```

#### **Key Correlations:**

* **Sex vs Survived** shows strong positive correlation (because female = 1 survives more).
* **Pclass** is negatively correlated with survival
  → Lower class (3) = lower survival
* **Fare** correlates positively with survival
  → Higher-paying passengers survived more
* **FamilySize & IsAlone** show moderate relations, confirming social factors.

---

## **6. Summary of Findings (Based ONLY on the EDA performed)**

✔ **Females survived at a much higher rate than males.**
✔ **Passenger class strongly affected survival** (1st class > 2nd > 3rd).
✔ **Younger passengers** had better survival chances.
✔ **Passengers traveling alone** had lower survival rates.
✔ **Fare and socio-economic status** played a role in survival.
✔ The extracted **Title** feature revealed social structure, useful for deeper analysis.

---

## **7. Conclusion**

* Handling missing values
* Basic visualization
* Simple feature engineering
* Understanding relationships through plots
* Identifying survival patterns
