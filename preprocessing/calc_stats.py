#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 18 10:22:13 2026

@author: jacob
"""
#%%Imports
import json
import csv

#%%Main process
def calc_stats(path):
    stats = open(path + 'stats.txt', 'w')
    #%%Make SHAK kode translator
    
    with open(path + 'Data/registers/nyfoedte.csv') as f:
        d = csv.reader(f)
        
        headers = next(d)
        
        n_reg = {}
        n_hos = {}
        
        for i, header in enumerate(headers):
            if header == 'AnsvarligRegion_Geo_Tekst':
                reg_txt = i
            elif header == 'AnsvarligInstitution_Kode':
                hos_kode = i
            elif header == 'AnsvarligInstitution_Tekst':
                hos_txt = i
        
        translator = {'1501': ['Kbh Amts Sygehus i Gentofte', 'Region Hovedstaden'],
                      '4212': ['OUH Svenborg Sygehus', 'Region Syddanmark'],
                      '7601': ['Viborg Sygehus', 'Region Midtjylland'],
                      '7002': ['Silkeborg Centralsygehus', 'Region Midtjylland'],
                      '1401': ['Frederiksberg Hosdpital', 'Region Hovedstaden'],
                      '1502': ['Kbh. Amts Sygehus i Glostrup', 'Region Hovedstaden'],
                      '7026': ['Skejby Sygehus', 'Region Midtjylland'],
                      '6501': ['Holstebro Centralsygehus', 'Region Midtjylland'],
                      '5001': ['Sønderborg Sygehus', 'Region Syddanmark'],
                      '5002': ['Haderslev Sygehus', 'Region Syddanmark'],
                      '6502': ['Herning Sygehus', 'Region Midtjylland']}
        
        for line in d:
            if line[hos_kode] not in translator.keys():
                translator[line[hos_kode]] = [line[hos_txt], line[reg_txt]]
    
    
    #%%Count births
    with open(path + 'Data/registers/combined.csv') as f:
        d = csv.reader(f)
    
        headers = next(d)
       
        births = sum(1 for line in d)
        stats.write('--Total Births--\n')
        stats.write('Total births in SDS: ' + str(births) + '\n')
    
    
    #%%Count DB
    with open(path + 'Data/image_data/img_data.json') as f:
        d = json.load(f)
    
    births_in_sql = len(d)
    
    stats.write('Total births in SQL db: '+ str(births_in_sql) + '\n')
    
    with open(path + 'Data/logs/birth_missing.csv') as f:
        d = csv.reader(f)
        headers = next(d)
        
        for i, head in enumerate(headers):
            if head == 'SHAK':
                shak_i = i
            elif head == 'birthdate':
                birth_i = i
        
        errors = {}
        missing = 0
        for row in d:
            missing += 1
            key = row[2]
            try:
                _, reg = translator[row[shak_i]]
            except:
                reg = 'NO SHAK'
            if key not in errors.keys():
                errors[key] = {}
                errors[key]['count'] = 1
                errors[key]['region'] = {}
                errors[key]['date'] = {}
            else:
                errors[key]['count'] += 1
            if row[birth_i][0:4] not in errors[key]['date'].keys():
                errors[key]['date'][row[birth_i][0:4]] = 1
            else:
                errors[key]['date'][row[birth_i][0:4]] += 1
            if reg not in errors[key]['region'].keys():
                errors[key]['region'][reg] = 1
            else:
                errors[key]['region'][reg] += 1

                
        stats.write('Total births missing in SQL db: ' + str(missing) + '\n')
        for key in errors:
            stats.write('\t- ' + str(key) + ': ' + str(errors[key]['count']) + '\n')
            stats.write('\t--Regional breakdown--\n')
            for reg in errors[key]['region']:
                stats.write('\t\t- ' + str(reg) + ': ' + str(errors[key]['region'][reg]) + '\n')
            stats.write('\t--Yearly breakdown--\n')
            dates = list(errors[key]['date'].keys())
            dates.sort()
            for date in dates:
                stats.write('\t\t- ' + str(date) + ': ' + str(errors[key]['date'][date]) + '\n')
            
        uacc = births - births_in_sql - missing
        stats.write('Unaccounted for: ' + str(uacc) + '\n')
    
    #%%Region count for missing imgs
    with open(path + 'Data/logs/birth_missing.csv') as f:
        d = csv.reader(f)
        headers = next(d)
        
        
    #%%Count births with cervix
    stats.write('\n')
    with open(path + 'Data/traindata.json') as f:
        d = json.load(f)
        n_train = len(d)
    with open(path + 'Data/testdata.json') as f:
        d = json.load(f)
        n_test = len(d) 
    with open(path + 'Data/traindata_SP.json') as f:
        d = json.load(f)
        n_train_SP = len(d)
    with open(path + 'Data/testdata_SP.json') as f:
        d = json.load(f)
        n_test_SP = len(d)
    cervix_births = n_train + n_test
    stats.write('Total births with cervix scans: ' + str(n_train + n_test) + '\n')
    stats.write('\t- Train/Test: ' + str(n_train) + ' / ' + str(n_test) + '\n')
    stats.write('Total births with cervix scans + SP: ' + str(n_train_SP + n_test_SP) + '\n')
    stats.write('\t- Train/Test: ' + str(n_train_SP) + ' / ' + str(n_test_SP) + '\n')

    #%%Count images
    stats.write('\n')
    stats.write('--Images--\n')
    
    with open(path + 'Data/image_data/misc/image_list.csv') as f:
        d = csv.reader(f)
        
        headers = next(d)
        
        imgs = sum(1 for line in d)
        
        stats.write('Total images: ' + str(imgs) + '\n')
    
    #%%Count cervix images
    all_count = {}
    SP_count = {}
    stats.write('\n')
    n_max = 0
    with open(path + 'Data/traindata.json') as f:
        d = json.load(f)
        n_train = 0
        for key in d.keys():
            n_train += len(d[key]['imgs'])
            if len(d[key]['imgs']) > n_max:
                n_max = len(d[key]['imgs'])
            if d[key]['Hospital'] in all_count.keys():
                all_count[d[key]['Hospital']] += 1
            else:
                all_count[d[key]['Hospital']] = 1

    with open(path + 'Data/testdata.json') as f:
        d = json.load(f)
        n_test = 0
        for key in d.keys():
            n_test += len(d[key]['imgs'])
            if len(d[key]['imgs']) > n_max:
                n_max = len(d[key]['imgs'])
            if d[key]['Hospital'] in all_count.keys():
                all_count[d[key]['Hospital']] += 1
            else:
                all_count[d[key]['Hospital']] = 1

    with open(path + 'Data/traindata_SP.json') as f:
        d = json.load(f)
        n_train_SP = 0
        for key in d.keys():
            n_train_SP += len(d[key]['imgs'])
            if d[key]['Hospital'] in SP_count.keys():
                SP_count[d[key]['Hospital']] += 1
            else:
                SP_count[d[key]['Hospital']] = 1

    with open(path + 'Data/testdata_SP.json') as f:
        d = json.load(f)
        n_test_SP = 0
        for key in d.keys():
            n_test_SP += len(d[key]['imgs'])
            if d[key]['Hospital'] in SP_count.keys():
                SP_count[d[key]['Hospital']] += 1
            else:
                SP_count[d[key]['Hospital']] = 1
    
    stats.write('Total cervix images: ' + str(n_train + n_test) + '\n')
    stats.write('\t- Train/Test: ' + str(n_train) + ' / ' + str(n_test) + '\n')
    stats.write('Total cervix images with SP: ' + str(n_train_SP + n_test_SP) + '\n')
    stats.write('\t- Train/Test: ' + str(n_train_SP) + ' / ' + str(n_test_SP) + '\n')
    stats.write('\n')  
    stats.write('Max number of cervix images in 1 birth: ' + str(n_max) + '\n')
    stats.write('Avg number of cervix images per birth: ' + str(round((n_train + n_test)/cervix_births,2)) + '\n')

    #%%Count errors
    with open(path + 'Data/logs/errors.csv') as f:
        d = csv.reader(f)
        headers = next(d)
        
        counter = {}
        
        for row in d:
            if row[1] not in counter.keys():
                counter[row[1]] = 1
            else:
                counter[row[1]] += 1
                
        for key in counter:
            stats.write('\t- ' + str(key) + ': ' + str(counter[key]) + '\n')       
    
    
    #%%Regional + hospital breakdown
    stats.write('\n')
    stats.write('--Regional + Hospital breakdown (births)--\n')
    
    with open(path + 'Data/image_data/img_data.json') as f:
        cprs = json.load(f)
    
    cpr_child = set(cprs.keys())
    
    with open(path + 'Data/registers/combined.csv') as f:
        d = csv.reader(f)
        
        _ = next(d)
        
        for line in d:
            if line[0] in cpr_child:
                try:
                    hos, reg = translator[line[4]]
                except:
                    reg = 'No SHAK Code'
                    hos = 'No SHAK Code'
                if reg not in n_reg.keys():
                    n_reg[reg] = 1
                else:
                    n_reg[reg] += 1
                if hos not in n_hos.keys():
                    n_hos[hos] = 1
                else:
                    n_hos[hos] += 1
       
    n_reg_cer = {}
    n_hos_cer = {}
    n_reg_cer_SP = {}
    n_hos_cer_SP = {}
    
        
    for key in all_count:
        try:
            hos, reg = translator[key]
        except:
            reg = 'No SHAK Code'
            hos = 'No SHAK Code'        
        if reg not in n_reg_cer.keys():
            n_reg_cer[reg] = all_count[key]
        else:
            n_reg_cer[reg] += all_count[key]
        if hos not in n_hos_cer.keys():
            n_hos_cer[hos] = all_count[key]
        else:
            n_hos_cer[hos] += all_count[key]

    for key in SP_count:
        try:
            hos, reg = translator[key]
        except:
            reg = 'No SHAK Code'
            hos = 'No SHAK Code'        
        if reg not in n_reg_cer_SP.keys():
            n_reg_cer_SP[reg] = SP_count[key]
        else:
            n_reg_cer_SP[reg] += SP_count[key]
        if hos not in n_hos_cer_SP.keys():
            n_hos_cer_SP[hos] = SP_count[key]
        else:
            n_hos_cer_SP[hos] += SP_count[key]
                        
    stats.write('\tRegions (Total/Cervix/Cervix + SP):\n')
    total = [0,0,0]
    for key in n_reg:
        count = [0,0,0]
        count[0] += n_reg[key]
        total[0] += n_reg[key]
        if key in n_reg_cer.keys():
            count[1] += n_reg_cer[key]
            total[1] += n_reg_cer[key]
        if key in n_reg_cer_SP.keys():
            count[2] += n_reg_cer_SP[key]
            total[2] += n_reg_cer_SP[key]
        if key == 'No SHAK Code':
            s_shak = str(count[0]) + ' / ' + str(count[1]) + ' / ' + str(count[2]) + '\n'
            continue
        else:
            s = str(count[0]) + ' / ' + str(count[1]) + ' / ' + str(count[2]) + '\n'
            stats.write('\t- ' + str(key) + ': ' + s)
    stats.write('\t- ' + 'No SHAK Code' + ': ' + s_shak)
    s_total = str(total[0]) + ' / ' + str(total[1]) + ' / ' + str(total[2]) + '\n'
    stats.write('\t- ' + 'TOTAL' + ': ' + s_total)
    
    stats.write('\n')
    
    stats.write('\tHospitals (Total/Cervix/Cervix + SP):\n')
    total = [0,0,0]
    for key in n_hos:
        count = [0,0,0]
        count[0] += n_hos[key]
        total[0] += n_hos[key]
        if key in n_hos_cer.keys():
            count[1] += n_hos_cer[key]
            total[1] += n_hos_cer[key]
        if key in n_hos_cer_SP.keys():
            count[2] += n_hos_cer_SP[key]
            total[2] += n_hos_cer_SP[key]
        if key == 'No SHAK Code':
            s_shak = str(count[0]) + ' / ' + str(count[1]) + ' / ' + str(count[2]) + '\n'
            continue
        else:
            s = str(count[0]) + ' / ' + str(count[1]) + ' / ' + str(count[2]) + '\n'
            stats.write('\t- ' + str(key) + ': ' + s)
    stats.write('\t- ' + 'No SHAK Code' + ': ' + s_shak)
    s_total = str(total[0]) + ' / ' + str(total[1]) + ' / ' + str(total[2]) + '\n'
    stats.write('\t- ' + 'TOTAL' + ': ' + s_total)
    
    #%%Scanner breakdown
    stats.write('\n')
    stats.write('--Scanner breakdown (images)--\n')
    

#%%Make script individual callable
if __name__ ==  '__main__':
    path = '/projects/users/data/UCPH/DeepFetal/projects/preterm/'
    calc_stats(path)
