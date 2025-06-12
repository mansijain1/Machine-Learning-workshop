from sklearn.preprocessing import StandardScaler

def preprocess_data(df):
    features = ['Fresh', 'Milk', 'Grocery', 'Frozen', 'Detergents_Paper', 'Delicassen']
    X = df[features].copy()
    X = X.dropna() 
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, X
