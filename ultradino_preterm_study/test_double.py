import os
import torch
from glob import glob

from sklearn.metrics import (accuracy_score,
                                roc_auc_score, recall_score)
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

from ultradino_preterm_study.utils.bias_utils import *


def test(model, loader_test, split_index, out_dir):

    # Load data
    units_columns, axis, df = load_data()
    #df = df.set_index('image_dir_clean')
    df = df[df[split_index]=='test']
    df = df.reset_index(drop=True)  

    df_sono = df.copy()

    os.environ["CUDA_VISIBLE_DEVICES"]= str(1)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  
    experiment_path = out_dir + '/test_results'
    os.makedirs(experiment_path, exist_ok=True)
    
    model.eval()
    model.to(device)

    test_predictions, test_logits, test_predictions_img, test_predictions_mask, test_predictions_land, test_labels, test_dir_clean = [], [], [], [], [], [], []

    for i, batch in enumerate(loader_test):

        for key,value in batch.items():
            if torch.is_tensor(value):
                batch[key]=value.to(device)

        with torch.no_grad():
            
            img_embedding, mask_embedding, pred_enc_img, pred_enc_mask, pred_enc_landmarks, outputs = model(batch)
            
            logits = outputs['preterm']
            logits = logits.squeeze(-1)
            preds = torch.sigmoid(logits)
            
            test_predictions.append(preds)
            test_logits.append(logits)
            
            labels = batch['label'].float() # Is it a list? Answer:  
            dir_clean = batch["image_dir_clean"] # The path to the image, to merge with meta-info later. Is it a list of strings? 
          
            logits = pred_enc_img.squeeze(-1)
            preds = torch.sigmoid(logits)
            test_predictions_img.append(preds)
            
            logits = pred_enc_mask.squeeze(-1)
            preds = torch.sigmoid(logits)
            test_predictions_mask.append(preds)
            
            logits = pred_enc_landmarks.squeeze(-1)
            preds = torch.sigmoid(logits)
            test_predictions_land.append(preds)
            
            
            test_labels.append(labels)
            test_dir_clean.append(dir_clean) #
                
    test_predictions = torch.concat(test_predictions).detach().cpu().numpy()
    test_predictions_img = torch.concat(test_predictions_img).detach().cpu().numpy()
    test_predictions_mask = torch.concat(test_predictions_mask).detach().cpu().numpy()
    test_predictions_land = torch.concat(test_predictions_land).detach().cpu().numpy()
    test_labels = torch.concat(test_labels).unsqueeze(-1).detach().cpu().numpy()
    test_dir_clean = [item for sublist in test_dir_clean for item in sublist] #No need to detach here?  Answer: No, it's a list of strings.
    test_raw_predictions = test_predictions.copy()

    test_predictions[test_predictions>0.5] = 1
    test_predictions[test_predictions<=0.5] = 0
    test_predictions = test_predictions.astype(int)
    
    for test_raw in [test_raw_predictions, test_predictions_img, test_predictions_mask, test_predictions_land]:
        auroc = roc_auc_score(test_labels, test_raw)
        print(auroc)
    auroc = roc_auc_score(test_labels, test_raw_predictions)   
    acc = accuracy_score(test_labels, test_predictions)
    rec = recall_score(test_labels, test_predictions, pos_label=1)
    spec = recall_score(test_labels, test_predictions, pos_label=0)
    sens = sensibility_at_specificity_85(test_labels, test_raw_predictions, target_specificity=0.85, tolerance=0.005)
    
    print(f'AUC: {np.round(auroc, 3)}, ACC: {np.round(acc, 3)}, SEN 85: {np.round(sens, 3)}, SPE: {np.round(spec, 3)}')

    df_metrics = pd.DataFrame({'AUC': [auroc], 'Sens85': [sens], 'Acc': [acc], 'Split': [split_index]})
    df_metrics.to_csv(f'{experiment_path}/validation_metrics_{split_index}.csv', index=False)
    # Now let's replace in df the confidence and predictions according to image path
    df = df.drop(['prediction', 'confidence'], axis=1)
    
    df = df.merge(pd.DataFrame({'image_dir_clean': test_dir_clean, 
                                'prediction': test_predictions.reshape(-1), 
                                'confidence': test_raw_predictions.reshape(-1),
                                'confidence_img': test_predictions_img.reshape(-1),
                                'confidence_mask': test_predictions_mask.reshape(-1),
                                'confidence_land': test_predictions_land.reshape(-1)}), on='image_dir_clean',
                                how='left')

    print(df)
    df['label'] = df['label'].astype(int)
    df.to_csv(f'{experiment_path}/split_{split_index}.csv', index=False)


    # To plot a general bias analysis
    df_in1, df_in2 = df.copy(), df_sono.copy()
    #df_in1['confidence'] = 0.5*df_in1['confidence'].values + 0.25*(df_in1['confidence_img'].values + df_in1['confidence_mask'].values)
    path_save = experiment_path + '/bias_analysis'
    os.makedirs(path_save, exist_ok=True)
    bias_analysis("UltraDINO", df_in1, axis,units_columns, path_save)

    double_bias_analysis('Sensibility', 'UltraDINO', df_in1, 'DL', df_in2, axis, units_columns, path_save)
    
    ## Courbe ROC Brute
    
    ## Courbe ROC Brute
    
    fpr_sono, tpr_sono, _ = roc_curve(df_sono['label'].values, df_sono['confidence'].values)
    roc_auc_sono = auc(fpr_sono, tpr_sono)
    
    fpr, tpr, _ = roc_curve(test_labels, test_raw_predictions)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(6, 6))
    
    plt.plot(fpr, tpr, color="blue", label=f'ROC UltraDINO (AUC = {roc_auc:.2f})')
    plt.plot(fpr_sono, tpr_sono, color="red", label=f'ROC SA-SONO (AUC = {roc_auc_sono:.2f})')
    
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.title('ROC Curves - Comparison')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    save_path = os.path.join(path_save, f'roc_overal_DINOvsSONO.png')
    plt.savefig(save_path, bbox_inches='tight')
    
    return df_in1, df_in2

def test_all(df_ultra, df_sono, out_dir):
    
    units_columns, axis, df = load_data()
    
    df_sono = df.copy()
    
    df_sono = df_sono[df_sono.image_dir_clean.isin(df_ultra.image_dir_clean)]
    df_ultra = df_ultra[df_ultra.image_dir_clean.isin(df_sono.image_dir_clean)]
                                    
                
    test_predictions = df_ultra['prediction'].values
    test_labels = df_ultra['label'].values
    test_raw_predictions = df_ultra['confidence'].values
    test_dir_clean = df_ultra['image_dir_clean'].values
    
    df = df_sono.copy()
     
    df = df.drop(['prediction', 'confidence'], axis=1)
    
    df_ultra = df.merge(pd.DataFrame({'image_dir_clean': test_dir_clean, 
                                'prediction': test_predictions.reshape(-1), 
                                'confidence': test_raw_predictions.reshape(-1)}), on='image_dir_clean',
                                how='left')
                                                               
    experiment_path = out_dir + '/test_results'
    os.makedirs(experiment_path, exist_ok=True)
                                
    auroc = roc_auc_score(test_labels, test_raw_predictions)
    acc = accuracy_score(test_labels, test_predictions)
    rec = recall_score(test_labels, test_predictions, pos_label=1)
    spec = recall_score(test_labels, test_predictions, pos_label=0)
    sens = sensibility_at_specificity_85(test_labels, test_raw_predictions, target_specificity=0.85, tolerance=0.005)
    
    print(f'AUC: {np.round(auroc, 3)}, ACC: {np.round(acc, 3)}, SEN 85: {np.round(sens, 3)}, SPE: {np.round(spec, 3)}')

    df_metrics = pd.DataFrame({'AUC': [auroc], 'Sens85': [sens], 'Acc': [acc], 'Split': ["All"]})
    df_metrics.to_csv(f'{experiment_path}/validation_metrics.csv', index=False)
    df_ultra.to_csv(f'{experiment_path}/split_all.csv', index=False)
    
    # To plot a general bias analysis
    df_in1, df_in2 = df_ultra.copy(), df_sono.copy()
    #df_in1['confidence'] = 0.5*(df_in1["confidence"].values + df_in2["confidence"].values)
    path_save = experiment_path + '/bias_analysis'
    os.makedirs(path_save, exist_ok=True)
    bias_analysis("UltraDINO", df_in1, axis,units_columns, path_save)
    print(axis)
    double_bias_analysis('Sensibility', 'UltraDINO', df_in1, 'DL', df_in2, axis, units_columns, path_save)
    
    ## Courbe ROC Brute
    
    fpr_sono, tpr_sono, _ = roc_curve(df_sono['label'].values, df_sono['confidence'].values)
    roc_auc_sono = auc(fpr_sono, tpr_sono)
    
    fpr, tpr, _ = roc_curve(test_labels, test_raw_predictions)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(6, 6))
    
    plt.plot(fpr, tpr, color="blue", label=f'ROC UltraDINO (AUC = {roc_auc:.2f})')
    plt.plot(fpr_sono, tpr_sono, color="red", label=f'ROC SA-SONO (AUC = {roc_auc_sono:.2f})')
    
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.title('Courbe ROC - modèle original')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    save_path = os.path.join(path_save, f'roc_overal_DINOvsSONO.png')
    plt.savefig(save_path, bbox_inches='tight')
    
    # To plot a general bias analysis
    df_in1, df_in2 = df_ultra.copy(), df_sono.copy()
    
    df_in1['confidence'] = (df_in1['confidence'].values + df_in2['confidence'].values) / 2
    #df_in2['confidence'] = -df_in2['expert_CL_max']
    
    path_save = experiment_path + '/bias_analysis_fusion'
    os.makedirs(path_save, exist_ok=True)
    bias_analysis("UltraDINO", df_in1, axis,units_columns, path_save)

    double_bias_analysis('Sensibility', 'UltraDINO+DL', df_in1, 'DL', df_in2, axis, units_columns, path_save)
    
    test_predictions = df_in1['prediction'].values
    test_labels = df_in1['label'].values
    test_raw_predictions = df_in1['confidence'].values
    test_dir_clean = df_in1['image_dir_clean'].values
                                
    auroc = roc_auc_score(test_labels, test_raw_predictions)
    acc = accuracy_score(test_labels, test_predictions)
    rec = recall_score(test_labels, test_predictions, pos_label=1)
    spec = recall_score(test_labels, test_predictions, pos_label=0)
    sens = sensibility_at_specificity_85(test_labels, test_raw_predictions, target_specificity=0.85, tolerance=0.005)
    
    print(f'AUC: {np.round(auroc, 3)}, ACC: {np.round(acc, 3)}, SEN 85: {np.round(sens, 3)}, SPE: {np.round(spec, 3)}')

    df_metrics = pd.DataFrame({'AUC': [auroc], 'Sens85': [sens], 'Acc': [acc], 'Split': ["All"]})
    df_metrics.to_csv(f'{experiment_path}/validation_metrics_ensemble.csv', index=False)
    
    
    ## Courbe ROC Brute
    
    fpr_sono, tpr_sono, _ = roc_curve(df_sono['label'].values, df_sono['confidence'].values)
    roc_auc_sono = auc(fpr_sono, tpr_sono)
    
    fpr, tpr, _ = roc_curve(df_in1['label'].values, df_in1['confidence'].values)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(6, 6))
    
    plt.plot(fpr, tpr, color="blue", label=f'ROC UltraDINO (AUC = {roc_auc:.2f})')
    plt.plot(fpr_sono, tpr_sono, color="red", label=f'ROC SA-SONO (AUC = {roc_auc_sono:.2f})')
    
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.title('Courbe ROC - modèle original')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    save_path = os.path.join(path_save, f'roc_overal_DINOvsSONO.png')
    plt.savefig(save_path, bbox_inches='tight')     
    
    
    
def test_from_pandas(path_predictions, out_dir, n_bins=5, font_size=24):
    
    units_columns, axis, df_sono = load_data(n_bins=n_bins)
    
    df_ultra = pd.read_csv(path_predictions)
    
    df_sono = df_sono[df_sono.image_dir_clean.isin(df_ultra.image_dir_clean)]
    df_ultra = df_ultra[df_ultra.image_dir_clean.isin(df_sono.image_dir_clean)]
                                    
    experiment_path = out_dir + '/test_results'
    os.makedirs(experiment_path, exist_ok=True)
                
    test_predictions = df_ultra['prediction'].values
    test_labels = df_ultra['label'].values
    test_raw_predictions = df_ultra['confidence'].values
    test_dir_clean = df_ultra['image_dir_clean'].values
    
    df = df_sono.copy()
     
    df = df.drop(['prediction', 'confidence'], axis=1)
    
    df_ultra = df.merge(pd.DataFrame({'image_dir_clean': test_dir_clean, 
                                'prediction': test_predictions.reshape(-1), 
                                'confidence': test_raw_predictions.reshape(-1)}), on='image_dir_clean',
                                how='left')
                                
    auroc = roc_auc_score(test_labels, test_raw_predictions)
    acc = accuracy_score(test_labels, test_predictions)
    rec = recall_score(test_labels, test_predictions, pos_label=1)
    spec = recall_score(test_labels, test_predictions, pos_label=0)
    sens = sensibility_at_specificity_85(test_labels, test_raw_predictions, target_specificity=0.85, tolerance=0.005)
    
    print(f'AUC: {np.round(auroc, 3)}, ACC: {np.round(acc, 3)}, SEN 85: {np.round(sens, 3)}, SPE: {np.round(spec, 3)}')

    df_metrics = pd.DataFrame({'AUC': [auroc], 'Sens85': [sens], 'Acc': [acc], 'Split': ["All"]})
    df_metrics.to_csv(f'{experiment_path}/validation_metrics.csv', index=False)
    df_ultra.to_csv(f'{experiment_path}/split_all.csv', index=False)
    
    # To plot a general bias analysis
    df_in1, df_in2 = df_ultra.copy(), df_sono.copy()
    #df_in2['confidence'] = -df_in2['expert_CL_max']
    path_save = experiment_path + '/bias_analysis'
    os.makedirs(path_save, exist_ok=True)
    bias_analysis("UltraDINO", df_in1, axis,units_columns, path_save)

    double_bias_analysis('Sensibility', 'UltraDINO', df_in1, 'DL', df_in2, axis, units_columns, path_save, font_size=font_size)
    
    ## Courbe ROC Brute
    
    fpr_sono, tpr_sono, _ = roc_curve(df_sono['label'].values, df_sono['confidence'].values)
    roc_auc_sono = auc(fpr_sono, tpr_sono)
    
    fpr, tpr, _ = roc_curve(test_labels, test_raw_predictions)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(6, 6))
    
    plt.plot(fpr, tpr, color="blue", label=f'ROC UltraDINO (AUC = {roc_auc:.2f})')
    plt.plot(fpr_sono, tpr_sono, color="red", label=f'ROC SA-SONO (AUC = {roc_auc_sono:.2f})')
    
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.title('Courbe ROC - modèle original')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    save_path = os.path.join(path_save, f'roc_overal_DINOvsSONO.png')
    plt.savefig(save_path, bbox_inches='tight')     
    
    # To plot a general bias analysis
    df_in1, df_in2 = df_ultra.copy(), df_sono.copy()
    
    df_in1['confidence'] = (df_in1['confidence'].values + df_in2['confidence'].values) / 2
    #df_in2['confidence'] = -df_in2['expert_CL_max']
    
    path_save = experiment_path + '/bias_analysis_fusion'
    os.makedirs(path_save, exist_ok=True)
    bias_analysis("UltraDINO", df_in1, axis,units_columns, path_save)

    double_bias_analysis('Sensibility', 'UltraDINO+DL', df_in1, 'DL', df_in2, axis, units_columns, path_save, font_size=font_size)
    
    test_predictions = df_in1['prediction'].values
    test_labels = df_in1['label'].values
    test_raw_predictions = df_in1['confidence'].values
    test_dir_clean = df_in1['image_dir_clean'].values
                                
    auroc = roc_auc_score(test_labels, test_raw_predictions)
    acc = accuracy_score(test_labels, test_predictions)
    rec = recall_score(test_labels, test_predictions, pos_label=1)
    spec = recall_score(test_labels, test_predictions, pos_label=0)
    sens = sensibility_at_specificity_85(test_labels, test_raw_predictions, target_specificity=0.85, tolerance=0.005)
    
    print(f'AUC: {np.round(auroc, 3)}, ACC: {np.round(acc, 3)}, SEN 85: {np.round(sens, 3)}, SPE: {np.round(spec, 3)}')

    df_metrics = pd.DataFrame({'AUC': [auroc], 'Sens85': [sens], 'Acc': [acc], 'Split': ["All"]})
    df_metrics.to_csv(f'{experiment_path}/validation_metrics_ensemble.csv', index=False)
    
    
    ## Courbe ROC Brute
    
    fpr_sono, tpr_sono, _ = roc_curve(df_sono['label'].values, df_sono['confidence'].values)
    roc_auc_sono = auc(fpr_sono, tpr_sono)
    
    fpr, tpr, _ = roc_curve(df_in1['label'].values, df_in1['confidence'].values)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(6, 6))
    
    plt.plot(fpr, tpr, color="blue", label=f'ROC UltraDINO (AUC = {roc_auc:.2f})')
    plt.plot(fpr_sono, tpr_sono, color="red", label=f'ROC SA-SONO (AUC = {roc_auc_sono:.2f})')
    
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.title('Courbe ROC - modèle original')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    save_path = os.path.join(path_save, f'roc_overal_DINOvsSONO.png')
    plt.savefig(save_path, bbox_inches='tight')     
    
    

