#segmentation = median_filter(segmentation, 5)
# Binary image, post-process the binary mask and compute labels
from skimage.measure import regionprops, label
from radiomics import featureextractor
import SimpleITK as sitk
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from statsmodels.multivariate.manova import MANOVA

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_predict
from sklearn.linear_model import LogisticRegressionCV

import pandas as pd
from neuroHarmonize import harmonizationLearn, harmonizationApply


from skimage.feature import local_binary_pattern
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import cv2

#data = [df_variables]


import os


def resample_cervix(image, masque, spacing, new_spacing):
    """
    Resample the image and mask to the new spacing.
    """
    w, h, c = image.shape
    
    w_ref, h_ref = int(w * spacing[0] / new_spacing[0]), int(h * spacing[1] / new_spacing[1])
    
    image_resampled = cv2.resize(image, (h_ref, w_ref), interpolation=cv2.INTER_CUBIC)
    masque_resampled = cv2.resize(masque, (h_ref, w_ref), interpolation=cv2.INTER_NEAREST)

    return image_resampled, masque_resampled


def resample_cervix(image, spacing, new_spacing):
    """
    Resample the image and mask to the new spacing.
    """
    w, h, c = image.shape
    
    w_ref, h_ref = int(w * spacing[0] / new_spacing[0]), int(h * spacing[1] / new_spacing[1])
    
    image_resampled = cv2.resize(image, (h_ref, w_ref), interpolation=cv2.INTER_CUBIC)

    return image_resampled
    
def load_segmentation(path_npy):
    
    seg = np.load(path_npy,allow_pickle=True)
    seg = seg.item(0)
    mask = np.argmax(seg['logits'],0) 
    
    return mask
    
from scipy.stats import pearsonr

def compute_correlations(df_normalized, target):

    pearsons = {}

    # Calcul pour chaque variable de la corrélation avec la cible
    for col in df_normalized.columns:
        
        try:
            a,b = df_normalized[col], target
            pearsons[col] = pearsonr(a.values, b)[0]
            #pearsons[col] = pearsonr(target.values, variable)[0]
        except:
            print(col)
            continue

        #Calcul des corrélations en valeur absolue
    df_pearson = pd.DataFrame(pearsons.values(), index=pearsons.keys())
    df_pearson['Abs'] = np.abs(df_pearson[0])

    # Classement pour chaque genre de caractéritiques:

    

    return df_pearson.sort_values('Abs', ascending=False)
    
def generate_results(features, harmonize=False):

    features_copied = features.copy()

    for feature in features_copied.keys():
          
        #try:
            continue
            path_save = os.path.join('Results/', feature)
            if not os.path.exists(path_save):
                os.makedirs(path_save)

            

            if feature != 'MOIs':
                 
                data_dict = [df.copy() for df in features_copied[feature]] #features_copied[feature].copy()
                for n, data in enumerate(data_dict):
                    
                    if n == 0:
                        kind = 'LBP'
                        rads = False
                        if not os.path.exists(os.path.join(path_save, kind)):
                            os.makedirs(os.path.join(path_save, kind))
                    else:
                        kind = 'Radiomics'
                        rads = True
                        if not os.path.exists(os.path.join(path_save, kind)):
                            os.makedirs(os.path.join(path_save, kind))


                    for model in ['logistic', 'knn', ]:
                        path_save_results = os.path.join(path_save, kind, model+'pca')
                        results, df_results = score_features(path_save_results, data, model, pca=True, rads =rads, harmonize=harmonize, plot=True)
                        results.to_csv(path_save_results+'.csv')
                        path_save_results = os.path.join(path_save, kind, model+'mda')
                        results, df_results = score_features(path_save_results, data, model, pca=False, rads =rads, harmonize=harmonize, plot=True)
                        results.to_csv(path_save_results+'.csv')

                data_all = pd.concat([feat for feat in data_dict],axis=1, join='inner')
                data = data_all.loc[:, ~data_all.T.duplicated()] 
                
                kind = 'Mix'
                rads = True
                if not os.path.exists(os.path.join(path_save, kind)):
                    os.makedirs(os.path.join(path_save, kind))


                for model in ['logistic', 'knn']:
                    path_save_results = os.path.join(path_save, kind, model+'pca')
                    results, df_results = score_features(path_save_results, data, model, pca=True, rads =rads, harmonize=harmonize, plot=True)
                    results.to_csv(path_save_results+'.csv')
                    path_save_results = os.path.join(path_save, kind, model+'mda')
                    results, df_results = score_features(path_save_results, data, model, pca=False, rads =rads, harmonize=harmonize, plot=True)
                    results.to_csv(path_save_results+'.csv')
                        
            else:
            
                data = features_copied[feature]
                kind = "Measures"
                rads = True
                if not os.path.exists(os.path.join(path_save, kind)):
                            os.makedirs(os.path.join(path_save, kind))
                
                for model in ['logistic', 'knn', 'rf']:
                        path_save_results = os.path.join(path_save, kind, model+'pca')
                        results, df_results = score_features(path_save_results, data, model, pca=True, rads =rads, harmonize=harmonize, plot=True)
                        results.to_csv(path_save_results+'.csv')
                        path_save_results = os.path.join(path_save, kind, model+'mda')
                        results, df_results = score_features(path_save_results, data, model, pca=False, rads =rads, harmonize=harmonize, plot=True)
                        results.to_csv(path_save_results+'.csv')
                        
                        
            
       # except:
       #    continue
    
    
    #ALL
    data_dict = [df.copy() for roi in ['Ant_Banos', 'Ant_Ext', 'Post_Ext', 'Cervix', 'Under_Post', 'Region_Out'] for df in features[roi]]
    data_dict = [df for df in data_dict if len(df) > 0]
    data_dict.append(features["MOIs"])
    data_all = pd.concat([feat for feat in data_dict],axis=1, join='inner')
    data = data_all.loc[:, ~data_all.T.duplicated()] 

    kind = 'Mix'
    path_save = os.path.join('Results/', "ALL")
    rads = True
    if not os.path.exists(os.path.join(path_save, kind)):
        os.makedirs(os.path.join(path_save, kind))


    for model in ['nn']:
        
        path_save_results = os.path.join(path_save, kind, model+'pca')
        results, df_results = score_features(path_save_results, data, model, pca=True, rads =rads, harmonize=harmonize, plot=True)
        results.to_csv(path_save_results+'.csv')
        df_results.to_csv(path_save_results+'pred.csv')
        path_save_results = os.path.join(path_save, kind, model+'mda')
        results, df_results = score_features(path_save_results, data, model, pca=False, rads =rads, harmonize=harmonize, plot=True)
        results.to_csv(path_save_results+'.csv')
        df_results.to_csv(path_save_results+'pred.csv')
        
        


def get_LBP(image, mask):
    # Paramètres LBP
    P = 16
    R = 1

    # Appliquer LBP à toute l'image
    lbp_image = local_binary_pattern(image, P, R, method='uniform')

    # Extraire les valeurs LBP uniquement dans la ROI
    lbp_roi = lbp_image[mask > 0]

    # Calculer l’histogramme des patterns dans la ROI
    n_bins = P + 2  # pour method='uniform' → nombre de bins possibles
    hist, _ = np.histogram(lbp_roi, bins=np.arange(0, n_bins + 1), density=True)
    
    hist = np.round(hist, 4)
    
    feat = {}
    for n in range(len(hist)):
         feat['LBP_' + str(n+1)] = hist[n]
    
    # hist = vecteur de features LBP (18 dimensions dans ton cas)
    return feat


def get_radiomics(image, segmentation_np):
#if True:
    # Convertir en image SimpleITK
    sitk_image = sitk.GetImageFromArray(segmentation_np)
    sitk_ct = sitk.GetImageFromArray(image)

    # Configurer l'extracteur avec 'shape2D' au lieu de 'shape'
    extractor = featureextractor.RadiomicsFeatureExtractor()
    extractor.enableAllFeatures()  # Active tous les descripteurs

    # Extraire les descripteurs
    features = extractor.execute(sitk_ct, sitk_image)

    # Filtrer pour afficher uniquement les caractéristiques de forme
    features = {k: v for k, v in features.items() if 'diagnostics' not in k}
    #print(features)
    return features
    
def normalize_img(array, mask_array): 

       roi_pixels = array[mask_array > 0]
       mean = np.mean(roi_pixels)
       std = np.std(roi_pixels)

       normalized_array = (array - mean) / std
        
       return normalized_array
       
       

def add_prefix_dict(prefix, dic):
    new_dic = {}
    for key in dic.keys():
        new_dic[prefix + '_' + key] = dic[key]
    return new_dic

def do_PCA(X):

    pca = PCA(n_components=0.995)
    X_pca = pca.fit_transform(X)

    print("Nombre de composantes retenues :", X_pca.shape[1])

    return X_pca




def encode_labels(targets):
    # Exemple
    y = targets
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    return y_encoded



def do_MANOVA(X_pca, y):
    # On crée le DataFrame avec les données PCA et les labels
    df = pd.DataFrame(X_pca, columns=[f'PC{i+1}' for i in range(X_pca.shape[1])])
    df['target'] = y  # Remplacer y_encoded par y

    # MANOVA
    manova = MANOVA.from_formula(' + '.join(df.columns[:-1]) + ' ~ target', data=df)

        # Récupérer les résultats du test MANOVA
    print(manova.mv_test())



def get_MDA(path_save, X, y, plot=False):

    lda = LDA(n_components=1)
    X_mda = lda.fit_transform(X, y)

    if plot:

        plt.hist(X_mda[y==0], bins=30, alpha=0.5, label='Class 0')
        plt.hist(X_mda[y==1], bins=30, alpha=0.5, label='Class 1')
        plt.xlabel('MDA projection')
        plt.ylabel('Frequency')
        plt.legend()
        plt.title('Distribution along Most Discriminant Axis (MDA)')
        plt.savefig(path_save + '_MDAPlot.png')
        #plt.show()
        

    return X_mda

def get_CV_auc(X, y, plot=False):
    # Cross-validated predicted probabilities
    clf = LogisticRegression(max_iter=1000)
    y_prob = cross_val_predict(clf, X, y, cv=10, method='predict_proba')[:, 1]

    fpr, tpr, thresholds = roc_curve(y, y_prob)
    roc_auc = auc(fpr, tpr)

    if plot:
        plt.figure(figsize=(6,6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (AUC = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC)')
        plt.legend(loc="lower right")
        plt.grid(True)
        #plt.show()

    return roc_auc

def scale_radiomics(df):

    df.replace([np.inf, -np.inf], 0, inplace=True)
    
    for col in df.columns:
        try:
            df[col] = df[col].astype(float)
        except:
            continue
    # Identifier les colonnes non numériques
    non_numeric_columns = df.select_dtypes(include=['object']).columns
    print("Colonnes non numériques :", non_numeric_columns)
    numeric_columns = df.select_dtypes(include=['number']).columns
    
    scaler = MinMaxScaler()

    normalized_data = scaler.fit_transform(df[numeric_columns])

    df = pd.DataFrame(normalized_data, columns = numeric_columns, index=df.index)
  
    return df
    




def harmonize_radiomics(data):

    site = pd.DataFrame(data['probe'])
    site = site.rename({'probe': 'SITE'},axis=1)

    X = data.drop(['label', 'probe'], axis=1)
    #X = X.apply(pd.to_numeric, errors="coerce')
    
    # Apprendre la correction ComBat
    model = harmonizationLearn(X.values, site)

    # Appliquer la correction sur les données
    X_harmonized = harmonizationApply(X.values, site, model[0])

    # Recréer un dataframe propre
    data_harmonized = pd.DataFrame(X_harmonized, columns=X.columns).fillna(0)
    
    data_harmonized.index = X.index
    X = data_harmonized
    print(X.shape)            # Dimensions
    print(np.isnan(X).sum().sum())  # Nombre de NaN
    print(np.isinf(X).sum().sum())  # Nombre d'infini
    print(np.all(X == 0).sum())     # Est-ce tout zéro ?
    print(np.var(X, axis=0)) 
    return data_harmonized


def score_features(path_save, data, model='logistic', pca=False, rads=False, harmonize=False, plot=False):

    y_encoded = encode_labels(data['label'])
    groups = data["cpr"]
    data = data.drop(['cpr'], axis=1)
    if rads:

        if harmonize:
            X = harmonize_radiomics(data)
            X = scale_radiomics(X)
        else:
            X = scale_radiomics(data)
            X = X.drop(['label'], axis=1)
    else:
        
        X = data.drop(['probe', 'label'], axis=1)
    
    #df_cor = compute_correlations(X, y_encoded)
    #df_cor.to_csv(os.path.join(path_save+'_corr.csv'))
    ind = X.index
    
    return score_MDA(path_save, ind, X, y_encoded, groups, model, pca=pca, plot=True)


def score_MDA(path_save, ind, X, y, groups, model, pca=False,plot=False):

    X_pca = do_PCA(X)
    
    #X_mda = get_MDA(path_save, X, y, plot=plot)

    #results = get_CV_auc(X_mda, y, plot=plot)

    
    if pca:
        results, df_results = get_CV_metrics(path_save, ind, X_pca, y, groups, model=model, cv=10, plot_roc=plot, auto_threshold=True)

    else:
        results, df_results = get_CV_metrics(path_save, ind, X, y, groups, model=model, cv=10, plot_roc=plot, auto_threshold=True)

    return results, df_results

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_predict, StratifiedKFold, StratifiedGroupKFold
from sklearn.metrics import roc_curve, auc, confusion_matrix, accuracy_score, recall_score, precision_score
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


from sklearn.neural_network import MLPClassifier

def get_CV_metrics(path_save, X, y, groups, model='logistic', cv=10, plot_roc=False, auto_threshold=True, optimize_k=True):

    skf = StratifiedGroupKFold(n_splits=cv, shuffle=True, random_state=42)
    n_classes = len(np.unique(y))
    print(model)
    if model == "nn":
        clf = MPLClassifier(hidden_layer_sizes=(64,5))
        
    elif model == 'logistic':
        clf = LogisticRegression(max_iter=1000)

    elif model == 'rf':
        clf = RandomForestClassifier(n_estimators=1000, n_jobs=-1, max_depth=30, max_features = None, min_samples_leaf=1, random_state=42)

    elif model == 'knn':
        # Optimisation de k si demandé
        if optimize_k:
            k_range = range(1, 31, 2)
            scores = []

            for k in k_range:
                clf_k = KNeighborsClassifier(n_neighbors=k)
                y_pred_k = cross_val_predict(clf_k, X, y, cv=skf, groups=groups)
                acc = accuracy_score(y, y_pred_k)
                scores.append(acc)
            print(scores)
            best_k = k_range[np.argmax(scores)]
            clf = KNeighborsClassifier(n_neighbors=best_k)
            print(f"✔️  Optimal k = {best_k} (Accuracy = {max(scores):.3f})")
        else:
            clf = KNeighborsClassifier(n_neighbors=5)  # Valeur par défaut

    else:
        raise ValueError("model must be 'logistic', 'knn', or 'rf'")

    # ======== Binaire =========
    if n_classes == 2:
        y_prob = cross_val_predict(clf, X, y, cv=skf,  groups=groups, method='predict_proba')[:, 1]
        
        fpr, tpr, thresholds = roc_curve(y, y_prob)
        roc_auc = auc(fpr, tpr)

        youden_index = tpr - fpr
        best_idx = np.argmax(youden_index)
        best_threshold = thresholds[best_idx]
        threshold = best_threshold if auto_threshold else 0.5
        
        y_pred = (y_prob >= threshold).astype(int)
        y_pred = cross_val_predict(clf, X, y, cv=skf, groups=groups)
        cm = confusion_matrix(y, y_pred)
        tn, fp, fn, tp = cm.ravel()

        accuracy = accuracy_score(y, y_pred)
        sensitivity = recall_score(y, y_pred)
        specificity = tn / (tn + fp)
        precision = precision_score(y, y_pred)

        if plot_roc:
            plt.figure(figsize=(6,6))
            plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (AUC = %0.2f)' % roc_auc)
            plt.scatter(fpr[best_idx], tpr[best_idx], marker='o', color='red', label='Best threshold = %.3f' % best_threshold)
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Receiver Operating Characteristic (ROC)')
            plt.legend(loc="lower right")
            plt.grid(True)
            plt.savefig(path_save + '.png')
            #plt.show()
            

        results = {
            'AUC': roc_auc,
            'Best threshold (Youden)': best_threshold,
            'Accuracy': accuracy,
            'Sensitivity (Recall)': sensitivity,
            'Specificity': specificity,
            'Precision': precision}
        

    # ======== Multi-classe =========
    else:
        y_pred = cross_val_predict(clf, X, y, cv=skf)
        accuracy = accuracy_score(y, y_pred)

        results = {'Accuracy': accuracy}

    
    return pd.Series(results)

def get_CV_metrics(path_save, ind, X, y, groups, model='logistic', cv=10, plot_roc=False, auto_threshold=True, optimize_k=True):

    skf = StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=42)
    n_classes = len(np.unique(y))
    
    if model == "nn":
        clf = MPLClassifier(hidden_layer_sizes=(64,5))
        
    elif model == 'logistic':
        clf = LogisticRegression(max_iter=3000)
        
    elif model == "LDA":
        clf = LDA(n_components=1)
    elif model == 'rf':
        clf = RandomForestClassifier(n_estimators=1000, n_jobs=-1, max_depth=30, max_features = None, min_samples_leaf=1, random_state=42)

    elif model == 'knn':
        # Optimisation de k si demandé
        if optimize_k:
            k_range = range(1, 100, 10)
            scores = []

            for k in k_range:
                clf_k = KNeighborsClassifier(n_neighbors=k)
                y_pred_k = cross_val_predict(clf_k, X, y, cv=skf, groups=groups)
                acc = accuracy_score(y, y_pred_k)
                scores.append(acc)
            print(scores)
            best_k = k_range[np.argmax(scores)]
            clf = KNeighborsClassifier(n_neighbors=best_k)
            print(f"✔️  Optimal k = {best_k} (Accuracy = {max(scores):.3f})")
        else:
            clf = KNeighborsClassifier(n_neighbors=5)  # Valeur par défaut

    else:
        raise ValueError("model must be 'logistic', 'knn', or 'rf'")

    # ======== Binaire =========
    if n_classes == 2:
        y_prob = cross_val_predict(clf, X, y, cv=skf, groups=groups, method='predict_proba')[:, 1]
        
        fpr, tpr, thresholds = roc_curve(y, y_prob)
        roc_auc = auc(fpr, tpr)

        youden_index = tpr - fpr
        best_idx = np.argmax(youden_index)
        best_threshold = thresholds[best_idx]
        threshold = best_threshold if auto_threshold else 0.5
        
        y_pred = (y_prob >= threshold).astype(int)
        y_pred = cross_val_predict(clf, X, y, cv=skf, groups=groups)
        cm = confusion_matrix(y, y_pred)
        tn, fp, fn, tp = cm.ravel()

        accuracy = accuracy_score(y, y_pred)
        sensitivity = recall_score(y, y_pred)
        specificity = tn / (tn + fp)
        precision = precision_score(y, y_pred)
        
        # Création du DataFrame avec les résultats
        df_results = pd.DataFrame({
            'prediction': y_pred,
            'confidence': y_prob,
            'label': y
        }, index=ind)
        
        if plot_roc:
            plt.figure(figsize=(6,6))
            plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (AUC = %0.2f)' % roc_auc)
            plt.scatter(fpr[best_idx], tpr[best_idx], marker='o', color='red', label='Best threshold = %.3f' % best_threshold)
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Receiver Operating Characteristic (ROC)')
            plt.legend(loc="lower right")
            plt.grid(True)
            plt.savefig(path_save + '.png')
            #plt.show()
            

        results = {
            'AUC': roc_auc,
            'Best threshold (Youden)': best_threshold,
            'Accuracy': accuracy,
            'Sensitivity (Recall)': sensitivity,
            'Specificity': specificity,
            'Precision': precision}
        

    # ======== Multi-classe =========
    else:
        y_pred = cross_val_predict(clf, X, y, cv=skf)
        accuracy = accuracy_score(y, y_pred)

         # Création du DataFrame avec les résultats
        df_results = pd.DataFrame({
            'prediction': y_pred,
            'confidence': [None] * len(y),  # Pas de score de confiance en multi-classe
            'label': y
        }, index=ind)


        results = {'Accuracy': accuracy}

    
    return pd.Series(results), df_results
