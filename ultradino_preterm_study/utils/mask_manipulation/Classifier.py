
from sklearn.preprocessing import MinMaxScaler

#data = [df_variables]

def scale_radiomics(df, index_col):


    df.index = df[index_col] ## A DEFINIR
    
   
    df.replace([np.inf, -np.inf], 0, inplace=True)
    
    # Identifier les colonnes non numériques
    non_numeric_columns = df.select_dtypes(include=['object']).columns
    print("Colonnes non numériques :", non_numeric_columns)
    numeric_columns = df.select_dtypes(include=['number']).columns
    
    scaler = MinMaxScaler()

    normalized_data = scaler.fit_transform(df[numeric_columns])

    df = pd.DataFrame(normalized_data, columns = numeric_columns, index=df.index)
  
    return df
    

from scipy.stats import pearsonr

def compute_correlations(df_normalized, target):

    pearsons = {}

    # Calcul pour chaque variable de la corrélation avec la cible
    for col in df_normalized.columns:
        
        try:
            variable = df_normalized[df_normalized.index.isin(target.index)][col].fillna(0).values
            
            df = pd.merge(df_normalized[df_normalized.index.isin(target.index)][col].fillna(0), target, 'inner', left_index=True, right_index=True)
            a,b = df[col], target
            pearsons[col] = pearsonr(a.values, b.values)[0]
            #pearsons[col] = pearsonr(target.values, variable)[0]
        except:
            print(col)
            continue

        #Calcul des corrélations en valeur absolue
    df_pearson = pd.DataFrame(pearsons.values(), index=pearsons.keys())
    df_pearson['Abs'] = np.abs(df_pearson[0])

    # Classement pour chaque genre de caractéritiques:

    print(df_pearson.loc[[col for col in df_pearson.index]].sort_values('Abs', ascending=False).head(30))

    return df_pearson
    
from sklearn.feature_selection import f_classif, SelectKBest

def select_variables(df_normalized, target, n_var=50):

    selector = SelectKBest(score_func=f_classif, k=n_var)
    X = df_normalized[df_normalized.index.isin(target.index)].fillna(0)

    # Fit selector and get selected feature names
    Xtr_array = selector.fit_transform(X, y)
    selected_features = X.columns[selector.get_support()]

    # Convert back to DataFrame
    Xtr = pd.DataFrame(Xtr_array, columns=selected_features, index=X.index)

    return Xtr
    
    
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from scipy.special import logit, expit


def train_classifier(df, target, folds, method='Logistic'):


    final_df = pd.DataFrame(columns= ['Case', 'predicted_score', 'label'])
    
    y = target

    X = df[df.index.isin(y.index)].fillna(0)
   # X = pd.merge(X, df_border[df_border.index.isin(y.index)].fillna(0)[cols], left_index=True, right_index=True)
    #X = X[X.index.isin(df_geometry.index)]
    #X['10pour'] = df_image['Image_original_firstorder_10Percentile']
    # #X['Image_Uncertainty ratio 0.05'] = df_image['Image_Uncertainty ratio 0.05']
    #X['Image_DSC_Uncertainty'] = df_image['Image_Uncertainty ratio 0.2']
    #X['std'] = df_image['Image_Uncertainty ratio 0.3']
    y = y[y.index.isin(X.index)].fillna(0)

   
    MAE = []

    for fold in folds:

        y_train = y[~X['Case'].isin([fold])]
        y_test = y[X['Case'].isin([fold])]
        X_train = X[~X['Case'].isin([fold])]
        X_test = X[X['Case'].isin([fold])]

        slice_names = list(X[X['Case'].isin([fold])].index)

        X_train = X_train.drop('Case', axis=1)
        X_test = X_test.drop('Case', axis=1)

        # === Feature Scaling (important for SVM) ===
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # === Train Regression Model ===
        if method == 'Logistic':
            model = LogisticRegression()
        else:
            
            model = RandomForestClassifier(n_estimators=100, random_state=42)

        model.fit(X_train, y_train)

        # === Predictions ===
        Y_pred = model.predict(X_test)

        final_df = pd.concat([final_df, pd.DataFrame(columns= ['Case', 'predicted_score', 'label'], data = zip(slice_names, Y_pred, y_test))], ignore_index=True)
    
    return final_df
    
 
  
# Choix de la cible
#target = df_scores['Mean_Lesion'].dropna()
