df.replace({'yes': 1, 'no': 0}, inplace=True)

# # One-hot encode categorical variables
# categorical_columns = ['job', 'marital', 'education', 'contact', 'month', 'poutcome']
# df = pd.get_dummies(df, columns=categorical_columns, drop_first=True)
