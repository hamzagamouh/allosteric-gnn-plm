import json

HOME_FOLDER="/storage/praha1/home/hamzagamouh"
folds=[]
for fold_id in range(5):
    folds+=[json.load(open(f"{HOME_FOLDER}/ahoj-allosteric/src/method_2/train_unps_fold_{fold_id}.json","r"))]


for i in range(5):
    for j in range(5):
        print(i,j,len(folds[i]),len(folds[j]),"inter",len(set(folds[i]).intersection(set(folds[j])))*100/max(len(folds[i]),len(folds[j])))