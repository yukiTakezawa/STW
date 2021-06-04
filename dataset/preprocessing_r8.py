from mat4py import loadmat
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split

print("reading mat file...")
dataset = loadmat("r8-emd_tr_te3.mat")
print("done")

words = []
for w in (dataset["words_tr"]):
    words += w
words = list(set(words))

train_bow = np.zeros((len(dataset["ytr"]), len(words)))
train_label = []
test_bow = np.zeros((len(dataset["yte"]), len(words)))
test_label = []

count = 0
for idx in tqdm(range(len(dataset["BOW_xtr"])), desc="[train]"):
    
    train_label.append(dataset["ytr"][idx])
    
    for i in range(len(dataset["words_tr"][idx])):
        w = dataset["words_tr"][idx][i]
        
        if (type(w) != str) or (w not in words):
            continue
        train_bow[count][words.index(w)] = dataset["BOW_xtr"][idx][i]
    
    #normalize
    train_bow[count] /= sum(train_bow[count])
    count += 1

count = 0
for idx in tqdm(range(len(dataset["BOW_xte"])), desc="[test]"):
    
    test_label.append(dataset["yte"][idx])
    
    for i in range(len(dataset["words_te"][idx])):
        w = dataset["words_te"][idx][i]
        
        if (type(w) != str) or (w not in words):
            continue
        
        test_bow[count][words.index(w)] = dataset["BOW_xte"][idx][i]
    
    #normalize
    test_bow[count] /= sum(test_bow[count])
    count += 1


print("saving...")
train_df = pd.DataFrame(train_bow, columns=words)
train_label_df = pd.DataFrame(train_label, columns=["data_label"])
test_df = pd.DataFrame(test_bow, columns=words)
test_label_df = pd.DataFrame(test_label, columns=["data_label"])

train_df = pd.concat([train_df, train_label_df], axis=1)
test_df = pd.concat([test_df, test_label_df], axis=1)

train_df.to_csv("r8/r8_train_all.csv", index=False)
test_df.to_csv("r8/r8_test.csv", index=False)

X_train, X_test, y_train, y_test = train_test_split(train_bow, train_label, test_size=0.2)

train_df = pd.DataFrame(X_train, columns=words)
train_label_df = pd.DataFrame(y_train, columns=["data_label"])
eval_df = pd.DataFrame(X_test, columns=words)
eval_label_df = pd.DataFrame(y_test, columns=["data_label"])
train_df = pd.concat([train_df, train_label_df], axis=1)
eval_df = pd.concat([eval_df, eval_label_df], axis=1)
train_df.to_csv("r8/r8_train.csv", index=False)
eval_df.to_csv("r8/r8_eval.csv", index=False)
print("done")
