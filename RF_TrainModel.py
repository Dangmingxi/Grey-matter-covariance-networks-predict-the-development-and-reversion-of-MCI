'''
Author: Mingxi Dang
Date: 2022-07-05 11:45:18
Description: The prediction models of RF were established based on the baseline score maps of the covariance network.
'''

import nibabel as nib
import numpy as np
import os
from scipy import interp
from sklearn.metrics import roc_curve,auc
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
import csv
import pandas as pd
from sklearn import metrics

rootdir = r'E:\longotudinal_MRI-new\MRI_data\DMN_MCI'
list = os.listdir(rootdir) 
x = []
bool_arr = np.ones((61 * 73 * 61))
# (61, 73, 61)
for i in range(0,len(list)):
       data = os.path.join(rootdir,list[i])
       img = nib.load(data)
       img_arr = img.get_fdata()
       img_arr = img_arr.reshape(img.shape[0] * img.shape[1] * img.shape[2])
       new_bool_arr = img_arr > 0.0
       bool_arr = np.logical_and(bool_arr, new_bool_arr)
       x.append(img_arr)

for r in range(0,len(x)):
        for c in range(0,len(x[0])):
            if x[r][c] == 0:
                for r2 in range(0, len(x)):
                    x[r2][c] = 0
x = np.mat(x)
# print(x.shape)

idx = np.argwhere(np.all(x[..., :] == 0, axis=0))
x = np.delete(x, idx, axis=1)
# print(x.shape)

#Data standardization
scaler = StandardScaler()
x_std = scaler.fit_transform(x)
print(x_std.shape)

img_b=nib.load(r'E:\longitudinal_MRI\MRI_data\baseline\NC_pcc_top3_weightvoxel\wv_1_sNC_pcc2_01_0001_10CHENGUOKUN_smwp1s20100128_08_ZhangZJ_ZL_ChenGuoKun-0006-00001-000176-01.nii')
affine = img_b.affine

data_label = np.genfromtxt(r'E:\longotudinal_MRI-new\RF/mci_label.csv', delimiter=',')
y = data_label[:, 0].astype(int)

#cross validation
loo = LeaveOneOut()
cv = KFold(n_splits=5, shuffle=True, random_state=100)

#Adjust parameters "n_estimators" according to the learning curve
scorel = []
for i in range(0,200,10):
    rfc = RandomForestClassifier(n_estimators=i+1,
                                 n_jobs=-1,
                                 random_state=90)
    score = cross_val_score(rfc,x_std,y,scoring='roc_auc', cv = KFold(n_splits=5, shuffle=True, random_state=100)).mean()
    scorel.append(score)
print(max(scorel),(scorel.index(max(scorel))*10)+1)
plt.figure(figsize=[20,5])
plt.plot(range(1,201,10),scorel)
plt.show()


# Further refine the learning curve within the defined boundaries
# scorel = []
# for i in range(90, 130):
#     rfc = RandomForestClassifier(n_estimators=i,
#                                  n_jobs=-1,
#                                  random_state=90)
#     score = cross_val_score(rfc, x_std, y, scoring='roc_auc', cv = KFold(n_splits=5, shuffle=True, random_state=100)).mean()
#     scorel.append(score)
# print(max(scorel),([*range(90,130)][scorel.index(max(scorel))]))
# plt.figure(figsize=[10, 40])
# plt.plot(range(90,130), scorel)
# plt.show()


#Adjust parameters "max_depth" 
scorel = []
for i in range(2, 22):
    rfc = RandomForestClassifier(n_estimators=108,
                                 random_state=90,
                                 max_depth=i)
    score = cross_val_score(rfc, x_std, y, scoring='roc_auc', cv = KFold(n_splits=5, shuffle=True, random_state=100)).mean()
    scorel.append(score)
print(max(scorel),([*range(2,22)][scorel.index(max(scorel))]))
plt.figure(figsize=[20, 5])
plt.plot(range(2, 22), scorel)
plt.show()


#Adjust parameters "max_features" 
scorel = []
for i in range(1,1000,2):
    rfc = RandomForestClassifier(n_estimators=108,
                                 random_state=90,
                                 max_depth=3,
                                 max_features=i)
    score = cross_val_score(rfc,x_std,y,scoring='roc_auc', cv = KFold(n_splits=5, shuffle=True, random_state=100)).mean()
    scorel.append(score)
print(max(scorel),(scorel.index(max(scorel))*2)+1)
plt.figure(figsize=[20,5])
plt.plot(range(1,1000,2),scorel)
plt.show()


#Adjust parameters "min_samples_leaf" 
param_grid = {'min_samples_leaf':np.arange(1,21)}

rfc = RandomForestClassifier(n_estimators=61,
                             random_state=90,
                             max_features=7
                             )
GS = GridSearchCV(rfc,param_grid,cv=loo)
GS.fit(x_std,y)
print(GS.best_params_)
print(GS.best_score_)

#Adjust parameters "min_samples_split" 
param_grid = {'min_samples_split':np.arange(2,22)}

rfc = RandomForestClassifier(n_estimators=61,
                             random_state=90,
                             max_features=7
                             )
GS = GridSearchCV(rfc,param_grid,cv=loo)
GS.fit(x_std,y)
print(GS.best_params_)
print(GS.best_score_)


#Adjust Criterion
param_grid = {'criterion':['gini', 'entropy']}
rfc = RandomForestClassifier(n_estimators=23,
                             random_state=90,
                             max_depth=3,
                             max_features= 25,
                             min_samples_leaf=2,
                             min_samples_split=5
                             )
GS = GridSearchCV(rfc,param_grid,scoring='roc_auc', cv = KFold(n_splits=5, shuffle=True, random_state=100))
GS.fit(x_std,y)
print(GS.best_params_)
print(GS.best_score_)



#Train model
rfc = RandomForestClassifier(n_estimators=108,
                                 random_state=90,
                                 max_depth=3,
                                 max_features= 81,
                                 min_samples_split=4
                             )

tprs = []
aucs = []
scorel = []
probas = []
y_true = []
feature_importances = []


mean_fpr = np.linspace(0, 1, 100)
probas = np.empty((0, 2))
y_true = np.empty((0,))
for train, test in cv.split(x_std):
    rfc.fit(x_std[train], y[train])
    probas_ = rfc.predict_proba(x_std[test])
    probas = np.vstack((probas, probas_))

    feature = rfc.feature_importances_
    feature_importances.append(feature)
    a = y[test]
    y_true = np.hstack((y_true, y[test]))
    # decision = rfc.decision_function(x_std[test])
    fpr, tpr, thresholds = roc_curve(y[test], probas_[:, 1])
    tprs.append(interp(mean_fpr, fpr, tpr))
    tprs[-1][0] = 0.0
    roc_auc = auc(fpr, tpr)
    aucs.append(roc_auc)

    score = rfc.score(x_std[test], y[test])
    scorel.append(score)

mean_score = np.mean(scorel)
print('acc:', mean_score)

mean_auc1 = np.mean(aucs)
print("auc1:", mean_auc1)

mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
std_auc = np.std(aucs)
print("mean_auc: %0.3f (+/- %0.3f)" % (mean_auc, std_auc))


tpr_minus_fpr = mean_tpr - mean_fpr
pos = np.argmax(tpr_minus_fpr) 
print("sensitivity", mean_tpr[pos])
print("specificity", 1 - mean_fpr[pos])
out = open(r'E:\longotudinal_MRI-new\RF\auc_result/mci_dmn_roc.csv', 'a', newline='')
csv_write = csv.writer(out, dialect='excel')
csv_write.writerow(mean_fpr)
csv_write.writerow(mean_tpr)
csv_write.writerow(y_true)
csv_write.writerow(probas[:, 1])


score_auc = cross_val_score(rfc, x_std, y, scoring='roc_auc',
                            cv=KFold(n_splits=5, shuffle=True, random_state=100))
print("test_auc: %0.3f (+/- %0.3f)" % (score_auc.mean(), score_auc.std()))


score_acc = cross_val_score(rfc, x_std, y, scoring='accuracy',
                            cv=KFold(n_splits=5, shuffle=True, random_state=100))
print("test_acc: %0.3f (+/- %0.3f)" % (score_acc.mean(), score_acc.std()))



feature_importances = np.array(feature_importances)
feature_importances_mean = []
for i in range(feature_importances.shape[1]):
    feature_importances_mean.append(np.mean(feature_importances[:, i]))
feature_importances_mean = np.array(feature_importances_mean)


tmp_img_arr = np.zeros((61 * 73 * 61))
tmp_img_arr[bool_arr] = feature_importances_mean

feature_importances_mean = tmp_img_arr.reshape(61,73,61)
feature_image = nib.Nifti1Image(feature_importances_mean, affine)
nib.save(feature_image, 'E:\longotudinal_MRI-new\RF\\auc_result/mci_dmn_feature.nii.gz')