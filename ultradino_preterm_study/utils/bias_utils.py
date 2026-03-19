import os

import pandas as pd
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.metrics import roc_curve, auc


from glob import glob
import natsort 
from scipy.stats import mannwhitneyu

from matplotlib.offsetbox import TextArea, AnnotationBbox

import re

import math
import shutil

from PIL import Image

from scipy.stats import gaussian_kde

#os.environ["KALIEDO_ARGS"] = "--disable-gpu --no-sandbox"
#pio.kaleido.scope = None  # reset any previous instance


def load_data(n_bins=5):
   
    # Load extra-meta-information
    meta_sono = sorted(glob('/data/proto/Aasa_PTB/results_clinical_paper_v2/dataset/logs_img_px_mask_version_*', recursive=True))
    dfs = []
    meta_sono = [pd.read_csv(csv) for csv in meta_sono]
    df_meta = pd.concat(meta_sono).reset_index()
    
    # Load CL read clinical values 
    path_cl = '/data/proto/Joris/Data/Base_de_base/base_paraskevas_expertCL_v2.csv'
    df_cl = pd.read_csv(path_cl, index_col=0)
    df_cl['angle'] = np.abs(df_cl['angle']) #Exclude calipers with vertical direction
    df_cl = df_cl[df_cl['angle'] < 70] #Exclude calipers with vertical direction
    df_cl['dicom_dir'] = df_cl['path_dcm']


    # Load predictions from Paraskevas
    results_sono = sorted(glob('/data/proto/sPTB-SA-SonoNet/results/sa-sononet/*', recursive=True))
    dfs = []
    results_sono = [pd.read_csv(csv) for csv in results_sono]
    df = pd.concat(results_sono).reset_index()
    
    path_size = "/data/proto/Joris/Data/Resampled_Cervix/df_MICCAI_updated.csv"
    df_size = pd.read_csv(path_size, index_col=0)
    
    df = pd.merge(df, df_size[['dicom_dir', 'size_h']], on='dicom_dir')
    
    mapping = {
        'asiatisk': 'Asian',
        'kaukasisk': 'Cauc.',
        'orientalsk': 'Mid. East',
        'afrokaribisk': 'African',
        'andet': 'Other'
    }

    df_meta['Etnicitet_mor'] = df_meta['Etnicitet_mor'].replace(mapping)
    
    df['Ethnicity'] = df_meta['Etnicitet_mor']
    df['Parity'] = df_meta['Paritet'].astype("Int64")
    # Rename the columns to clean names to make the plots, define units, stratify continuous columns into stratified groups
    df['CL>25'] = df['cervical_length'].apply(lambda el: el > 25)
    df['CL>25'] = df['CL>25'].apply(lambda el: 'CL > 25mm' if el == True else 'CL < 25mm')
        
    df.rename(columns={'size_h': 'Depth', 'trimester': "Trimester", 'bmi': 'BMI','ga_in_weeks': 'GA', 
           'weeks_to_birth': 'Time to Birth', 'device_name': 'Device Name', 'birth_weight': "Birth Weight",
           'placenta_weight': "Placenta Weight",'year_of_birth': "Birth Year", 'pixel_spacing': "Pixel Spacing",
           'cervical_length': "CL", 'age': "Mother Age", 'CL>25': 'CL Category', 'birthplace': "Birth Place"}, inplace=True)
           
    units_columns = {
        "Depth": "mm",
        "Ethnicity": "",
        "Paritet": "",
        "Trimester": "",
        "BMI": "", #"kg/m²",
        "GA": "weeks",
        "Weeks to Birth": "weeks",
        "Device Name": "",  # chaîne de caractères (nom du dispositif)
        "Birth Weight": "kg",
        "Placenta Weight": "g",
        "Birth Year": "",
        "Pixel Spacing": "μm",#/pixel",
        "CL": "mm",
        "Mother Age": "",
        "CL Category": ""  # étiquette ou classe (ex: "normale", "courte", etc.)
    }

    axis = ["BMI",  "Mother Age","Ethnicity", "Parity",  "CL", 'GA', "Birth Weight", "Birth Place", "Birth Year",'Device Name', "Pixel Spacing", 'Depth']
    
    
    df['Birth Place'] = df['Birth Place'].astype(str)
    df['Pixel Spacing'] = 10000*df['physical_delta_x']
    df['Birth Weight'] = round(df['Birth Weight']/1000, 1)
    df['Depth'] = df['Depth']*10
    #df = df[df['Depth'] > 112]

    names = {}
    for n, place in enumerate(df['Birth Place'].unique()):
        names[place] = 'H. ' + str(n+1)
        
    df['Birth Place'] = df['Birth Place'].apply(lambda el: names[el])
    df['CL_base'] = df['CL']
    
    #df = pd.merge(df, df_cl, on='dicom_dir')
    #df['CL'] = df['expert_CL_max']
    
    df['Pixel Spacing Class'] = df['Pixel Spacing']
    df['Pixel Spacing Class'] = categorize_numeric_column(df, 'Pixel Spacing Class', n_bins=n_bins, method='quantile',
                                                        new_col_name='Pixel Spacing Class', unit=units_columns['Pixel Spacing'], dropna=True)
   # return units_columns, axis, df#, df_meta
    for axe in axis:
            #print(axe)
            if not is_column_string(df, axe):
                
                axe_type = detect_column_type(df, axe, unique_threshold=10)
                if axe == 'Pixel Spacing':
                    n_bins=n_bins
                else:
                    n_bins=n_bins
                if axe_type == "numérique":
                    df[axe] = categorize_numeric_column(df, axe, n_bins=n_bins, method='quantile',
                                                        new_col_name=axe, unit=units_columns[axe], dropna=True)
                                                        
    return units_columns, axis, df#, df_meta
   


def plot_subgroup_composition(df_in, axis, path_save, n_bins=5):

    df = df_in.copy()
    sd, mean = {}, {}
    bias_dict = {}
    
    axis.append('label')
    
    dataset_visualization_parameters = {}
    dataset_visualization_parameters['columns_to_plot'] = axis
    dataset_visualization_parameters['titles'] = {}

    for column in axis:
        dataset_visualization_parameters['titles'][column] = column
               
                                                  
    for axe in axis:
    
        if not is_column_string(df, axe):
            axe_type = detect_column_type(df, axe, unique_threshold=10)
            if axe_type == "numérique":
            
                if axe == 'Pixel Spacing':
                    n_bins=n_bins
                else:
                    n_bins=n_bins
                df[axe] = categorize_numeric_column(df, axe, n_bins=n_bins, method='quantile',
                                                    new_col_name=axe, unit=units[axe], dropna=True)
        df_axe = df[df[axe].notna()].copy()
        group_counts = df_axe[axe].dropna().value_counts()
        valid_groups = group_counts[group_counts >= 30].index
        sorted_groups = sorted(valid_groups, key=extract_numeric_start)
        
        os.makedirs(os.path.join(path_save,axe), exist_ok=True)
        
        for subgroup in sorted_groups:
            subgroup = str(subgroup)
            df_mask = df_axe[df_axe[axe] == subgroup]
            
                # Output directory for individual images
            temp_dir = os.path.join(path_save,axe,'output_images')
            output_path = os.path.join(path_save,axe,subgroup+'.png')

            # Plot and save images
            image_paths = plot_and_save_pie_charts_or_density(df_mask, dataset_visualization_parameters, temp_dir=temp_dir, output_path=output_path,
                                                              show_percentages=False, slice_font_size=18, title_font_size=25, 
                                                              chart_size=400, overall_title="", overall_title_size=30)
   
def bias_analysis(model, df_in, axis, units, path_save):

    df = df_in.copy()
    sd, mean = {}, {}
    bias_dict = {}
    
    for axe in axis:
        if not is_column_string(df, axe):
            axe_type = detect_column_type(df, axe, unique_threshold=10)
            if axe_type == "numérique":
            
                if axe == 'Pixel Spacing':
                    n_bins=5
                else:
                    n_bins=5
                df[axe] = categorize_numeric_column(df, axe, n_bins=n_bins, method='quantile',
                                                    new_col_name=axe, unit=units[axe], dropna=True)
        df_axe = df[df[axe].notna()].copy()
        os.makedirs(path_save, exist_ok=True)

       # bias_plots(df_axe, model, axe, path_save_axe)
       # sd[axe], mean[axe] = roc_bias(df_axe, model, axe, path_save_axe)
        res = roc_bias_min_max(df_axe, model, col=axe)
        if res:
           bias_dict[axe] = res
        
    plot_bias_radar_by_quadrant(bias_dict, model, "Detection of Bias" ,path_save)
    
    return sd, mean

def double_bias_analysis(kind, model_1, df_1, model_2, df_2, axis, units, path_save, font_size=24):

    df1, df2 = df_1.copy(), df_2.copy()

    image_paths = []
    for axe in axis:

        for df in [df1,df2]:

            if not is_column_string(df, axe):
                
                axe_type = detect_column_type(df, axe, unique_threshold=10)
                if axe == 'Pixel Spacing':
                    n_bins=5
                else:
                    n_bins=5
                if axe_type == "numérique":
                    df[axe] = categorize_numeric_column(df, axe, n_bins=n_bins, method='quantile',
                                                        new_col_name=axe, unit=units[axe], dropna=True)
                                                        
        df1_axe, df2_axe = df1[df1[axe].notna()].copy(), df2[df2[axe].notna()].copy()
        
        if axe in ['Parity']:
            df1_axe[axe] = df1_axe[axe].astype(str)
            df2_axe[axe] = df2_axe[axe].astype(str)
            
        res1, boot1 = roc_bias(kind, df1_axe, col=axe)
        res2, boot2 = roc_bias(kind, df2_axe, col=axe) # Dictionnaire de sensibilités pour chaque groupe
        vars_list = df1_axe[axe].dropna().unique()
        
        if axe in ['Parity', 'Birth Place']:
            vars_list = natsort.natsorted(vars_list)
            
        elif not type(vars_list) == np.ndarray:
            vars_list = df1_axe[axe].unique().categories
           # print(vars_list)
            
        vars_list = list(vars_list)
        vars_list = [var for var in vars_list if var in res1.keys()]
        
        os.makedirs(path_save, exist_ok=True)

       # bias_plots(df_axe, model, axe, path_save_axe)
       # sd[axe], mean[axe] = roc_bias(df_axe, model, axe, path_save_axe)
        
        cardinal = dict(df1_axe[axe].value_counts())
        try:
             path = plot_double_bias_radar_by_quadrant(kind, axe, vars_list, cardinal, res1,res2, boot1, boot2, model_1,model_2, path_save, font_size=font_size)
             image_paths.append(path)
        except:
             continue
       
    concat_images(path_save, image_paths, os.path.join(path_save, 'concat.png'), images_per_row=4)#, chart_size=400)
    
    return 



def extract_numeric_start(group_label):
    """
    Extrait le premier entier d'un label de groupe (ex: '10 to 20 kg' -> 10)
    """
    match = re.match(r"(\d+)", str(group_label))
    return int(match.group(1)) if match else float('inf')

def is_column_string(df, col, threshold=0.8):
    fraction_str = df[col].apply(lambda x: isinstance(x, str)).mean()
    return fraction_str >= threshold
    
def detect_column_type(df, col, unique_threshold=15):
    # Nombre de valeurs uniques
    n_unique = df[col].nunique(dropna=True)
    # Vérifie si toutes les valeurs sont entières (même si type float)
    is_all_integer = pd.api.types.is_integer_dtype(df[col]) or \
                     all(df[col].dropna() == df[col].dropna().astype(int))
    
    # Règle simple
    if n_unique <= unique_threshold and is_all_integer:
        return 'catégorique'
    elif pd.api.types.is_bool_dtype(df[col]) and n_unique <= unique_threshold:
        return 'catégorique (booléen)'
    elif n_unique <= unique_threshold and pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_categorical_dtype(df[col]):
        return 'catégorique (objet/catégorie)'
    else:
        return 'numérique'

def categorize_numeric_column(df, col, n_bins=3, method='quantile', new_col_name=None, unit='', dropna=True):
    """
    Catégorise une colonne numérique en groupes avec intervalles entiers et unité.

    Args:
        df (pd.DataFrame): DataFrame source
        col (str): nom de la colonne numérique à catégoriser
        n_bins (int): nombre de catégories désirées
        method (str): 'quantile' ou 'uniform'
        new_col_name (str): nom de la nouvelle colonne (par défaut col+'_group')
        unit (str): unité à afficher à la fin de l'intervalle, ex. 'kg'
        dropna (bool): si True, les NaN seront ignorés dans les bins

    Returns:
        pd.Series: nouvelle colonne catégorisée avec labels texte
    """
    if new_col_name is None:
        new_col_name = col + '_group'

    if method not in ['quantile', 'uniform']:
        raise ValueError("method must be 'quantile' or 'uniform'")

    # Supprimer les NaN temporairement pour construire les intervalles
    data = df[col].dropna() if dropna else df[col]

    # Crée les bins
    if method == 'quantile':
        bins = pd.qcut(data, q=n_bins, duplicates='drop')
    else:  # uniform
        bins = pd.cut(data, bins=n_bins)

    # Génère les labels avec entiers et unité
    labels = []
    for interval in bins.cat.categories:
        left = int(round(interval.left))
        right = int(round(interval.right))
        labels.append(f"{left}-{right}{unit}".strip())

    # Applique les nouveaux labels
    if method == 'quantile':
        categorized = pd.qcut(df[col], q=n_bins, labels=labels, duplicates='drop')
    else:
        categorized = pd.cut(df[col], bins=n_bins, labels=labels)

    # Retourne la colonne catégorisée (sans l'ajouter au df sauf si tu veux le faire toi-même)
    return categorized

def categorize_numeric_column(df, col, n_bins=3, method='quantile', new_col_name=None, unit='', dropna=True):
    """
    Catégorise une colonne numérique en groupes avec intervalles entiers et unité,
    en générant des labels personnalisés selon la variable (col).

    Args:
        df (pd.DataFrame): DataFrame source
        col (str): nom de la colonne numérique à catégoriser
        n_bins (int): nombre de catégories désirées
        method (str): 'quantile' ou 'uniform'
        new_col_name (str): nom de la nouvelle colonne (par défaut col+'_group')
        unit (str): unité à afficher à la fin de l'intervalle, ex. 'kg'
        dropna (bool): si True, les NaN seront ignorés dans les bins

    Returns:
        pd.Series: nouvelle colonne catégorisée avec labels texte
    """

    import pandas as pd
    import numpy as np

    if new_col_name is None:
        new_col_name = col + '_group'

    if method not in ['quantile', 'uniform']:
        raise ValueError("method must be 'quantile' or 'uniform'")

    # Supprimer les NaN temporairement pour construire les intervalles
    data = df[col].dropna() if dropna else df[col]

    # Crée les bins
    if method == 'quantile':
        bins = pd.qcut(data, q=n_bins, duplicates='drop')
    else:  # uniform
        bins = pd.cut(data, bins=n_bins)

    intervals = bins.cat.categories
    
    # Fonction pour générer labels personnalisés selon la variable
    def get_labels_for_col(col, intervals):
        labels = []
        # Labels standard pour certains variables
        simple_labels = ["Very Low", "Low", "Medium", "High", "Very High"]

        #if col in ["Pixel Spacing Class"]:
            # Utiliser les labels simples, adapter selon le nb de bins
        #    labels = simple_labels[:len(intervals)]

        if col in ["Birth Weight"]:
            # kg typiquement, afficher <x kg, x-y kg, >z kg
            for i, interval in enumerate(intervals):
                left = round(interval.left, 1)
                right = round(interval.right, 1)
                if i == 0:
                    labels.append(f"<{right}{unit}")
                elif i == len(intervals) - 1:
                    labels.append(f">{left}{unit}")
                else:
                    labels.append(f"{left}-{right}{unit}")


        elif col in ["GA", "CL", "Mother Age", "Pixel Spacing", "Pixel Spacing Class"]:#, "BMI", "Pixel Spacing"]:   # Gestational Age en semaines
            for i, interval in enumerate(intervals):
                left = round(interval.left)
                right = round(interval.right)
                if i == 0:
                    labels.append(f"<{right}{unit}")
                elif i == len(intervals) - 1:
                    labels.append(f">{left}{unit}")
                else:
                    labels.append(f"{left}-{right}{unit}")


        elif col == "Device Name":
            # C'est une variable catégorielle probablement, juste garder intervals textuels
            labels = [str(interval) for interval in intervals]

        elif col in ["Birth Year"]:
            # Année de naissance, labels avec années
            for i, interval in enumerate(intervals):
                left = int(interval.left)
                right = int(interval.right)
                if i == 0:
                    labels.append(f"≤{right}")
                elif i == len(intervals) - 1:
                    labels.append(f">{left}")
                else:
                    labels.append(f"{left}-{right}")

        elif col == "Birth Place":
            # Probablement catégoriel, labels textuels simples
            labels = [str(interval) for interval in intervals]

        else:
            # Par défaut, labels avec intervalles arrondis
            for interval in intervals:
                left = int(round(interval.left))
                right = int(round(interval.right))
                labels.append(f"{left}-{right}{unit}".strip())

        return labels

    labels = get_labels_for_col(col, intervals)
    
    # Applique les nouveaux labels
    if method == 'quantile':
        categorized = pd.qcut(df[col], q=n_bins, labels=labels, duplicates='drop')
    else:
        categorized = pd.cut(df[col], bins=n_bins, labels=labels)

    return categorized

def sensibility_at_specificity_85(y_true, y_score, target_specificity=0.85, tolerance=0.005):
    """
    Calcule la sensibilité (TPR) maximale pour une spécificité (1 - FPR) proche de 0.85,
    en autorisant une tolérance variable.

    Args:
        y_true: valeurs vraies (0 ou 1)
        y_score: scores de prédiction (probabilités)
        target_specificity: valeur cible pour la spécificité
        tolerance: marge d’erreur autorisée autour de la spécificité

    Returns:
        float: la plus grande sensibilité (TPR) parmi les points satisfaisant la condition
    """
    from sklearn.metrics import roc_curve
    import numpy as np

    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    specificity = 1 - fpr
    diffs = np.abs(specificity - target_specificity)

    close_enough = diffs <= tolerance
    
    # Ajuster la tolérance jusqu’à trouver au moins un point
    while not any(close_enough):
        tolerance += 0.005
       # print(tolerance)
        close_enough = diffs <= tolerance

    # Parmi les spécificités proches, on prend la plus grande TPR (sensibilité)
    tpr_candidates = tpr[close_enough]
    max_tpr = np.max(tpr_candidates)
    return max_tpr

def bootstrap_sensitivities(df, label_col, pred_col, n_iterations=100):
 
    sensitivities = []
  
    for _ in range(n_iterations):
 
        sample_df = df.sample(n=len(df), replace=True, random_state=None)
        y_true_group = sample_df[label_col]
        y_score_group = sample_df[pred_col]
        if len(y_true_group.unique()) > 1:
           sens = sensibility_at_specificity_85(y_true_group, y_score_group)
           sensitivities.append(sens)
        
    return sensitivities
    
def bootstrap_aucs(df, label_col, pred_col, n_iterations=100):

    aucs = []
    
    for _ in range(n_iterations):
    
        sample_df = df.sample(n=len(df), replace=True, random_state=None)
        y_true_group = sample_df[label_col]
        y_score_group = sample_df[pred_col]
        
        fpr, tpr, _ = roc_curve(y_true_group, y_score_group)
        AUC = round(auc(fpr, tpr), 2)
        aucs.append(AUC)
        
    return aucs
    
def roc_bias(kind, df, col, bootstrap=True, label_col='label', pred_col='confidence', max_curves=5):

 
    group_counts = df[col].dropna().value_counts()
    valid_groups = group_counts[group_counts >= 30].index
    sorted_groups = sorted(valid_groups, key=extract_numeric_start)

    specs = {}
    boot = {}
    
    for group in sorted_groups:
        
        mask = df[col] == group
        df_mask = df[df[col] == group]
        y_true_group = df.loc[mask, label_col]
        y_score_group = df.loc[mask, pred_col]

        if len(np.unique(y_true_group)) < 2:
            print(f"[Avertissement] Groupe '{group}' ignoré (label constant)")
            continue
        
        if kind == 'Sensibility':
            spec = sensibility_at_specificity_85(y_true_group, y_score_group)
            if not np.isnan(spec):
                specs[group] = round(spec, 2)
            if bootstrap:
                boot[group] = bootstrap_sensitivities(df_mask, label_col, pred_col, n_iterations=100)
            # L'idée consiste à calculer une suite de valeurs via bootstraping
        
        else:
            # On garde les courbes ROC (optionnel, visuel seulement)
            fpr, tpr, _ = roc_curve(y_true_group, y_score_group)
            specs[group] = round(auc(fpr, tpr), 2)
            if bootstrap:
                boot[group] = bootstrap_aucs(df_mask, label_col, pred_col, n_iterations=100)
        
    # Plot ROC global par groupes (inchangé)
    # ... tu peux garder le bloc de dessin ROC si utile ...

    return specs, boot

def roc_bias_min_max(df, model, col, label_col='label', pred_col='confidence', min_group_size=30):
    """
    Calcule les spécificités min et max à 85% de sensibilité pour chaque groupe d’un facteur.
    """
    group_counts = df[col].dropna().value_counts()
    valid_groups = group_counts[group_counts >= min_group_size].index
    sorted_groups = sorted(valid_groups, key=extract_numeric_start)
    specs_by_group = {}
    aucs = []
    boot = {}
    #print(col)
    for group in sorted_groups:
    
        mask = df[col] == group
        df_mask = df[df[col] == group]
        y_true = df.loc[mask, label_col]
        y_score = df.loc[mask, pred_col]

        if len(np.unique(y_true)) < 2:
            print(f"[Avertissement] Groupe '{group}' ignoré (label constant)")
            continue

        spec = sensibility_at_specificity_85(y_true, y_score)
        #print(group, "Spec ok", len(df_mask))
        boot[group] = bootstrap_sensitivities(df_mask, label_col, pred_col, n_iterations=40)
        #print("Bootstrap ok")
        if not np.isnan(spec):
            specs_by_group[group] = spec
        fpr, tpr, _ = roc_curve(y_true, y_score)
        
        aucs.append(auc(fpr, tpr))
       #if col == 'Birth Weight':
       #      print(group, spec, aucs[-1])
       #      print(np.abs(1 - fpr - .85))
    if not specs_by_group:
        return None  # Aucun groupe valide

    min_group = min(specs_by_group, key=specs_by_group.get)
    max_group = max(specs_by_group, key=specs_by_group.get)
    return {
        'min_spec': specs_by_group[min_group],
        'max_spec': specs_by_group[max_group],
        'group_min': min_group,
        'group_max': max_group,
        'min_boot': boot[min_group],
        'max_boot': boot[max_group]
    }
    

from matplotlib.offsetbox import TextArea, AnnotationBbox, VPacker
import re


def plot_bias_radar_by_quadrant(
    bias_dict,
    model,
    title="Biais par facteur",
    path_save=None
):
    import matplotlib.pyplot as plt
    import numpy as np
    import os

    plt.rcParams.update({
        "text.usetex": False,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman", "Helvetica Neue", "Times New Roman", "DejaVu Serif"],
        "axes.unicode_minus": False,
    })

    # Définir les groupes et ordre
    quadrants = {
        "Mère": {
            "vars": ["Ethnicity", "BMI", "Mother Age"],
            "range_deg": (0, 90),
            "color": "lavender"
        },
        "Bébé": {
            "vars": ["GA", "Birth Weight", 'CL'],
            "range_deg": (90, 180),
            "color": "lightblue"
        },
        "Imagerie": {
            "vars": ["Device Name", "Pixel Spacing"],
            "range_deg": (180, 270),
            "color": "mistyrose"
        },
        "Environnement": {
            "vars": ["Birth Year", "Birth Place"],
            "range_deg": (270, 360),
            "color": "honeydew"
        }
    }

    # Attribuer manuellement les angles à chaque variable
    angle_map = {}
    labels = []
    angles_deg = []
    for group in quadrants:
        vars_list = quadrants[group]["vars"]
        start, end = quadrants[group]["range_deg"]
        theta_range = np.linspace(start, end, len(vars_list) + 2)[1:-1]
        for n, (var, angle_deg) in enumerate(zip(vars_list, theta_range)):
            
            if len(vars_list) == 2:
                if n == 0:
                    angle_deg -= 90 / 16
                elif n == len(vars_list) - 1:
                    angle_deg += 90 / 20
            elif len(vars_list) == 3:
                if n == 0:
                    angle_deg -= 90 / 20
                elif n == len(vars_list) - 1:
                    angle_deg += 90 / 20
                        
            angle_rad = np.deg2rad(angle_deg)
            angle_map[var] = angle_rad
            # Ajout des groupes favorisés/défavorisés dans le label
            labels.append(f"{var}\n↑{bias_dict[var]['group_max']}\n↓{bias_dict[var]['group_min']}")
            angles_deg.append(angle_deg)

    # Compléter pour boucler le radar
    labels += [labels[0]]
    angles = [angle_map[lbl.split('\n')[0]] for lbl in labels[:-1]] + [angle_map[labels[0].split('\n')[0]]]

    # Spécificités min/max
    min_specs = [bias_dict[lbl.split('\n')[0]]["min_spec"] for lbl in labels[:-1]] + [bias_dict[labels[0].split('\n')[0]]["min_spec"]]
    max_specs = [bias_dict[lbl.split('\n')[0]]["max_spec"] for lbl in labels[:-1]] + [bias_dict[labels[0].split('\n')[0]]["max_spec"]]

    # Plot
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8])
    ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8'], color='gray', size=15)

    # 1. Ajout des axes verticaux et horizontaux
    ax.vlines(angles, 0, 1, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
    ax.hlines([0.2, 0.4, 0.6, 0.8, 1.0], 0, 2*np.pi, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)

    # 3. Colorier chaque quadrant manuellement avec des tons pastels doux
    quadrant_colors = [
        ("#d0e1f2", 0, np.pi/2),         # Bleu pastel doux
        ("#e6d6f0", np.pi/2, np.pi),     # Violet pastel doux
        ("#fbe2e5", np.pi, 3*np.pi/2),   # Rose pâle
        ("#d9f2e6", 3*np.pi/2, 2*np.pi)  # Vert menthe pâle
    ]
    
    for color, start, end in quadrant_colors:
        ax.fill_between(
            np.linspace(start, end, 100),
            0, 1, color=color, alpha=0.3
        )

  

    # Tracer les courbes
    ax.plot(angles, max_specs, color='teal', linewidth=2, label='Favored Group', alpha=0.5)
    ax.plot(angles, min_specs, color='crimson', linewidth=2, label='Unfavored Group', alpha=0.5)
    ax.fill(angles, max_specs, color='teal', alpha=0.1)
    ax.fill(angles, min_specs, color='crimson', alpha=0.1)

    # Tracer l'axe vertical complet (0° et 180°)
    ax.plot([0, 0], [0, 1.1], color='black', linewidth=1.5)       # Haut (0°)
    ax.plot([np.pi, np.pi], [0, 1.1], color='black', linewidth=1.5)  # Bas (180°)

    # Tracer l'axe horizontal complet (90° et 270°)
    ax.plot([np.pi/2, np.pi/2], [0, 1.1], color='black', linewidth=1.5)  # Gauche (90°)
    ax.plot([3*np.pi/2, 3*np.pi/2], [0, 1.1], color='black', linewidth=1.5)  # Droite (270°)
   
   
    ax.set_xticks([])
    # Ajout des étiquettes enrichies
    font_size=24
    for theta, label in zip(angles[:-1], labels[:-1]):
        parts = label.split('\n')
        stat, p = mannwhitneyu(bias_dict[parts[0]]['min_boot'], bias_dict[parts[0]]['max_boot'], alternative='two-sided')
        
        if len(parts) == 3:
            txt1 = TextArea(parts[0], textprops=dict(size=font_size, weight='bold'))
            if p < 0.05 and p > 0.001:
               txt2 = TextArea(parts[1] + '*', textprops=dict(size=font_size-4, color='teal'))     # Favored group
            elif p < 0.001:
               txt2 = TextArea(parts[1] + '**', textprops=dict(size=font_size-4, color='teal'))
            else:
               txt2 = TextArea(parts[1], textprops=dict(size=font_size-4, color='teal'))
           # txt2 = TextArea(parts[1], textprops=dict(size=font_size-4, color='teal'))     # Favored group
           
            txt3 = TextArea(parts[2], textprops=dict(size=font_size-4, color='crimson'))  # Unfavored group
            
            vpack = VPacker(children=[txt1, txt2, txt3], align="center", pad=0, sep=1)
            ab = AnnotationBbox(
                vpack,
                (theta, 1),  # angle en rad, rayon
                frameon=False,
                box_alignment=(0.5, 0.5)
            )
            ax.add_artist(ab)

    plt.subplots_adjust(left=0.25, right=0.75, top=0.85, bottom=0.15)
    ax.set_title(f'{model}', fontsize=28, pad=30)
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, -.25), fontsize=18)
    ax.grid(True, linestyle='--', linewidth=0.6, alpha=0.7)

    if path_save:
        
        save_path = os.path.join(path_save, f'bias_quadrant_radar_{model}.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"[Info] Radar plot sauvegardé : {save_path}")
    plt.close()
    
def plot_double_bias_radar_by_quadrant(kind, axe, vars_list, cardinal, res1,res2, boot1, boot2, model_1,model_2, path_save, font_size=24):
    
    import matplotlib.pyplot as plt
    import numpy as np
    import os

    plt.rcParams.update({
        "text.usetex": False,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman", "Helvetica Neue", "Times New Roman", "DejaVu Serif"],
        "axes.unicode_minus": False,
    })

   
    range_deg = (0, 360)
    theta_range = np.linspace(0, 360, len(vars_list) + 1)[:-1]
    # Attribuer manuellement les angles à chaque variable
    angle_map = {}
    labels = []
    angles_deg = []
    
    for var, angle_deg in zip(vars_list, theta_range):

        angle_rad = np.deg2rad(angle_deg)
        angle_map[var] = angle_rad
        # Ajout des groupes favorisés/défavorisés dans le label
        labels.append(f"{var}\n{res1[var]}\n{res2[var]}")
        angles_deg.append(angle_deg)
    
    
                  
    # Compléter pour boucler le radar
    labels += [labels[0]]
    angles = [angle_map[lbl.split('\n')[0]] for lbl in labels[:-1]] + [angle_map[labels[0].split('\n')[0]]]

    # Spécificités min/max
    specs_m1 = [res1[var] for var in vars_list] + [res1[vars_list[0]]]
    specs_m2 = [res2[var] for var in vars_list] + [res2[vars_list[0]]]

    # Plot
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    if kind == 'Sensibility':
        ax.set_ylim(0, 1)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8])
        ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8'], color='gray', size=15)
    else:
        ax.set_ylim(0.5, 1)
        ax.set_yticks([0.5, 0.6, 0.7, 0.8, 0.9])
        ax.set_yticklabels(['0.5', '0.6', '0.7', '0.8', '0.9'], color='gray', size=15)

    # 1. Ajout des axes verticaux et horizontaux
    if kind == 'Sensibility':
        ax.vlines(angles, 0, 1, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
        ax.hlines([0.2, 0.4, 0.6, 0.8, 1.0], 0, 2*np.pi, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
        # 1. Ajout des axes verticaux et horizontaux
    else:
        ax.vlines(angles, 0.5, 1, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
        ax.hlines([0.5, 0.6, 0.7, 0.8, 0.9, 1.0], 0, 2*np.pi, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
        
    # Tracer les courbes
    ax.plot(angles, specs_m1, color='teal', linewidth=2, label=model_1, alpha=0.5)
    ax.plot(angles, specs_m2, color='crimson', linewidth=2, label=model_2, alpha=0.5)
    ax.fill(angles, specs_m1, color='teal', alpha=0.1)
    ax.fill(angles, specs_m2, color='crimson', alpha=0.1)
   
    ax.set_xticks([])
    # Ajout des étiquettes enrichies
    
    if axe != 'Birth Place':
        font_size = font_size + 4 
    else:
        font_size = font_size 
    for theta, label in zip(angles[:-1], labels[:-1]):
        parts = label.split('\n')
        stat, p = mannwhitneyu(boot1[parts[0]], boot2[parts[0]], alternative='two-sided')
        
        if len(parts) == 3:
            txt1 = TextArea(parts[0], textprops=dict(size=font_size, weight='bold'))
            txt2 = TextArea('n=' + str(cardinal[parts[0]]), textprops=dict(size=font_size-4, color='black'))
            if p < 0.05 and p > 0.001:
               txt3 = TextArea(parts[1] + '*', textprops=dict(size=font_size-4, color='teal'))     # Favored group
            elif p < 0.001:
               txt3 = TextArea(parts[1] + '**', textprops=dict(size=font_size-4, color='teal'))
            else:
               txt3 = TextArea(parts[1], textprops=dict(size=font_size-4, color='teal'))
               
            txt4 = TextArea(parts[2], textprops=dict(size=font_size-4, color='crimson'))  # Unfavored group

            vpack = VPacker(children=[txt1, txt2, txt3, txt4], align="center", pad=0, sep=1)
            ab = AnnotationBbox(
                vpack,
                (theta, 1),  #angle en rad, rayon
                frameon=False,
                box_alignment=(0.5, 0.5)
            )
            ax.add_artist(ab)

    plt.subplots_adjust(left=0.25, right=0.75, top=0.85, bottom=0.15)
    ax.set_title(f'{axe}', fontsize=30, pad=60)
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, -.35), fontsize=18)
    ax.grid(True, linestyle='--', linewidth=0.6, alpha=0.7)
    # ax.plot([0, 0], [0, 1.1], color='black', linewidth=1.5)       # Haut (0°)
    if path_save:
        
        save_path = os.path.join(path_save, f'bias_quadrant_radar_{axe}.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"[Info] Radar plot sauvegardé : {save_path}")
        
    plt.close()
    return save_path




def plot_pie_charts_or_density_from_df(
    df, 
    columns, 
    title_dict, savepath,
    show_percentages=False, 
    slice_font_size=11, 
    title_font_size=16, 
    chart_size=4, 
    overall_title=None, 
    overall_title_size=20
):
    """
    Plots aligned pie charts or density curves based on the variable type using Matplotlib.
    Pie charts are used for categorical variables, and density curves for continuous variables.

    Parameters:
    - df: Pandas DataFrame containing the data.
    - columns: List of column names to plot.
    - title_dict: Dict mapping each column name to its title.
    - show_percentages: If True, show percentages in pie slices; else show labels.
    - slice_font_size: Font size for text inside pie slices.
    - title_font_size: Font size for individual plot titles.
    - chart_size: Size of each subplot (in inches).
    - overall_title: Optional string title above all plots.
    - overall_title_size: Font size of the overall title.
    """

    num_charts = len(columns)
    fig, axes = plt.subplots(3, num_charts//3, figsize=(12,14))
    axes = axes.flatten()
    # Ensure axes is iterable even for one plot
    if num_charts == 1:
        axes = [axes]

    color_palettes = [
        ['#636EFA', '#EF553B', '#00CC96', '#AB63FA'],
        ['#FFA07A', '#20B2AA', '#FF69B4', '#9370DB'],
        ['#FFD700', '#ADFF2F', '#FF4500', '#4682B4'],
        ['#8A2BE2', '#FF6347', '#3CB371', '#FFD700']
    ]

    for i, col in enumerate(columns):
        ax = axes[i]
        data = df[col].dropna().astype(str)
        fixed_orders = sorted(data.unique())
        
        if pd.api.types.is_numeric_dtype(data):
            # Continuous variable → density plot
                # Density
            x_vals = np.linspace(data.min(), data.max(), 200) 
            y_vals = gaussian_kde(data)(np.linspace(data.min(), data.max(), 200)) * len(data)
            ax.plot(x_vals, y_vals, color="#636EFA", lw=2)
            ax.fill_between(x_vals, 0, y_vals, color="#636EFA", alpha=0.3)

        else:

            # Suppose `fixed_orders[col]` is a dict with column → list of categories
            counts = data.value_counts().reindex(fixed_orders, fill_value=0)
            labels = counts.index.tolist()
            values = counts.values.tolist()

            colors = color_palettes[i % len(color_palettes)]

            if show_percentages:
                autopct = '%1.1f%%'
                labeldistance = 1.1
            else:
                autopct = None
                labeldistance = 1.2

            if show_percentages:
                wedges, texts, autotexts = ax.pie(
                    values,
                    labels=labels,
                    autopct='%1.1f%%',
                    colors=colors,
                    startangle=90,
                    textprops={'fontsize': slice_font_size},
                    wedgeprops={'edgecolor': 'black', 'linewidth': 1.5}
                )
            else:
                wedges, texts, autotexts = ax.pie(
                    values,
                    labels=None,
                    autopct='a',
                    colors=colors,
                    startangle=90,
                    textprops={'fontsize': slice_font_size},
                    wedgeprops={'edgecolor': 'black', 'linewidth': 1.5}
                )

            if show_percentages:
                for t in autotexts:
                    t.set_fontsize(slice_font_size)
            
            # Optional: write the actual labels instead of percentages
            for i, atext in enumerate(autotexts):
                atext.set_text(f"{labels[i]}")  # replace percentage with label
                atext.set_fontsize(slice_font_size)
    
            ax.set_title(title_dict.get(col, col), fontsize=title_font_size)

        ax.axis('equal')  # Make pies circular

    if overall_title:
        fig.suptitle(overall_title, fontsize=overall_title_size)

    #plt.tight_layout(rect=[0, 0, 1, 0.95] if overall_title else None)
    plt.savefig(savepath, dpi=100)
    plt.close()
    

def plot_and_save_pie_charts_or_density(df, dataset_visualization_parameters, temp_dir, output_path, show_percentages=True, slice_font_size=14, title_font_size=16, chart_size=300, overall_title=None, overall_title_size=20):
    """
    Plots and saves pie charts or density curves based on the variable type using Plotly.
    Saves each plot individually as an image, and later concatenates them into a grid.

    Parameters:
    - df: Pandas DataFrame containing the data.
    - columns: List of column names in the DataFrame to plot. 
    - title_dict: Dictionary with titles for each plot. Keys should match column names from the DataFrame.
    - output_dir: Directory to save the individual plot images.
    - show_percentages: Boolean, if True shows percentages in slices (for pie charts), else shows group names.
    - slice_font_size: Integer, font size for the text inside pie slices.
    - title_font_size: Integer, font size for the titles above plots.
    - chart_size: Integer, size of each plot (affects width and height).
    - overall_title: String, the main title to display above all plots.
    - overall_title_size: Integer, font size for the overall title.
    """
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)
    
    image_paths = []
    columns = dataset_visualization_parameters['columns_to_plot']
    title_dict = dataset_visualization_parameters['titles']

    plot_pie_charts_or_density_from_df(df, columns, title_dict, output_path)
        
           
    
def concat_images(output_dir, image_paths, output_path, images_per_row=4, chart_size=400):
    """
    Concatenate PNG images in a grid with a fixed number of images per row.

    Parameters:
    - output_dir: Temporary directory where individual PNGs are stored.
    - image_paths: List of paths to PNG images.
    - output_path: Path to save the final concatenated PNG image.
    - images_per_row: How many images per row.
    - chart_size: Final size (WxH) for each image in the grid.
    """

    num_images = len(image_paths)
    num_rows = math.ceil(num_images / images_per_row)

    # Create the final blank canvas (white background)
    total_width = images_per_row * chart_size
    total_height = num_rows * chart_size

    final_image = Image.new("RGB", (total_width, total_height), (255, 255, 255))

    for idx, img_path in enumerate(image_paths):
        img = Image.open(img_path).convert("RGB")

        # Resize to ensure uniform size
        img = img.resize((chart_size, chart_size), Image.LANCZOS)

        row = idx // images_per_row
        col = idx % images_per_row

        final_image.paste(img, (col * chart_size, row * chart_size))

    final_image.save(output_path, format="PNG")
    # shutil.rmtree(output_dir)  # if needed
