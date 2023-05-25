#========================= IMPORT PACKAGES ===========================
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
import re
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn import metrics

#==================== DATA SELECTION =========================

print("-----------------------------------------")
print("============ Data Selection =============")
print("-----------------------------------------")
data=pd.read_csv("Dataset.csv")
print(data.head(10))
print()

#================== PREPROCESSING =============================

#=== checking missing values ===

print("-----------------------------------------")
print("========= Checking missing values  ======")
print("-----------------------------------------")
print(data.isnull().sum())
print()

data.drop_duplicates(inplace = True)

#=== drop unwanted columns ===

print("----------------------------------------------")
print("============= Drop unwanted columns  =========")
print("----------------------------------------------")
print()
print("1.Before drop unwanted columns :",data.shape)
print()
print()
data_1=data.drop(['Unnamed: 0','Date'], axis = 1)
print("2.After drop unwanted columns  :",data_1.shape)
print()
print()


#========================= NLP TECHNIQUES ============================

#=== TEXT CLEANING ==== 

cleanup_re = re.compile('[^a-z]+')
def cleanup(sentence):
    sentence = str(sentence)
    sentence = sentence.lower()
    return sentence


print("----------------------------------------------")
print("============ Before Applying NLP  ============")
print("----------------------------------------------")
print()
print(data['Review_text'].head(10))

print("----------------------------------------------")
print("============ After Applying NLP  =============")
print("----------------------------------------------")
print()

data_1["Summary_Clean"] = data_1["Review_text"].apply(cleanup) 
data_1["URL"] = data_1["URL"].apply(cleanup) 

print(data_1["Summary_Clean"].head(10))

#========================= SENTIMENT ANALYSIS ==========================

#=== POS, NEG, NEUTRAL ===

analyzer = SentimentIntensityAnalyzer()
data_1['compound'] = [analyzer.polarity_scores(x)['compound'] for x in data_1['Summary_Clean']]
data_1['neg'] = [analyzer.polarity_scores(x)['neg'] for x in data_1['Summary_Clean']]
data_1['neu'] = [analyzer.polarity_scores(x)['neu'] for x in data_1['Summary_Clean']]
data_1['pos'] = [analyzer.polarity_scores(x)['pos'] for x in data_1['Summary_Clean']]


#======================= DATA SPLITTING ===========================

X = data_1["Summary_Clean"]
y = data_1['Rev_Type']
X_train, X_test,y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

print("==============================================")
print("---------------- Data Splitting --------------")
print("==============================================")
print()
print("Total No.of data's : ", data_1.shape[0])
print()
print("Total No.of training data's : ", X_train.shape[0])
print()
print("Total No.of testing data's : ", X_test.shape[0])



#================= VECTORIZATION ================================

vector = CountVectorizer(stop_words = 'english', lowercase = True)

#fitting the data
training_data = vector.fit_transform(X_train)

#tranform the test data
testing_data = vector.transform(X_test)   

print("==============================================")
print("---------------- Vectorization --------------")
print("==============================================")
print()
print(training_data)

#========================== CLASSIFICATION =================================

from sklearn.ensemble import AdaBoostClassifier


print("==============================================")
print("------------------- AdaBoost -----------------")
print("==============================================")
print()

#=== initialize the model ===
ada_boost = AdaBoostClassifier()

#=== fitting the model ===
ada_boost = ada_boost.fit(training_data, y_train)

#=== predict the model ===
y_pred_ada = ada_boost.predict(testing_data)

#=== PERFORMANCE ANALYSIS ===

cm=metrics.confusion_matrix(y_pred_ada,y_test)

TP=cm[0][0]
TN=cm[0][1]
FP=cm[1][0]
FN=cm[1][1]

Total=TP+TN+FP+FN

Acc_lr=(TP+FN+FP)/Total

print("============ Performance Analysis =========")
print()
print(" Accuracy :",Acc_lr*100,'%')
print()
print(metrics.classification_report(y_pred_ada,y_test))
print()


import matplotlib.pyplot as plt
import seaborn as sns
sns.heatmap(cm, annot=True)
plt.title("Adaboost")
plt.show()

# =========== LR ===

from sklearn import linear_model


print("==============================================")
print("------------------- LOGISTIC REGRESSION -----------------")
print("==============================================")
print()

#=== initialize the model ===
lr= linear_model.LogisticRegression()

#=== fitting the model ===
lr = lr.fit(training_data, y_train)

#=== predict the model ===
y_pred_ada = lr.predict(testing_data)

#=== PERFORMANCE ANALYSIS ===

cm=metrics.confusion_matrix(y_pred_ada,y_test)

TP=cm[0][0]
TN=cm[0][1]
FP=cm[1][0]
FN=cm[1][1]

Total=TP+TN+FP+FN

Acc=(TP+FN+FP)/Total

print("============ Performance Analysis =========")
print()
print(" Accuracy :",Acc*100,'%')
print()
print(metrics.classification_report(y_pred_ada,y_test))
print()


import matplotlib.pyplot as plt
import seaborn as sns
sns.heatmap(cm, annot=True)
plt.title("LR")
plt.show()

#========================== PREDICTION =================================

for i in range(0,10):
    if y_pred_ada[i]==0:
        print("==========================")
        print()
        print(" The review is fake ")
    else:
        print("==========================")
        print()
        print(" The review is real ")

prediction=4


a=prediction

data_label=data_1['Rev_Type']


x1=data_label
for i in range(0,len(data_1)):
    if x1[i]==a:
        idx=2  
    
    
data_frame1_c=data_1['Author']

Req_data_c=data_frame1_c[1]

print("Identified user = ",Req_data_c )

#========================== VISUALIZATION ==============================

#pie graph

plt.figure(figsize = (5,5))
counts =data_1['Rev_Type'].value_counts()
plt.pie(counts, labels = counts.index, startangle = 90, counterclock = False, wedgeprops = {'width' : 0.6},autopct='%1.1f%%', pctdistance = 0.55, textprops = {'color': 'black', 'fontsize' : 15}, shadow = True,colors = sns.color_palette("Paired")[3:])
plt.text(x = -0.35, y = 0, s = 'Total Reviews: {}'.format(data_1.shape[0]))
plt.title('Total No.of reviews', fontsize = 14);
plt.show()

# ===== Graphs =========

import matplotlib.pyplot as plt 

import seaborn as sns
sns.scatterplot(x=data_1['Rating'], y=data_1['neg'], hue=data_1['Rev_Type'])
plt.title("Scatter Plot")
plt.show()



#---

sns.barplot(y=[Acc_lr,Acc],x=['Adaboost','LR'])
plt.title("Comparison")
plt.show()

# ---

import matplotlib.pyplot as plt
plt.hist(y)
plt.show() 


# ---- 

fig, ax = plt.subplots(figsize=(6,6)) 
sns.heatmap(data_1[["Rating", "neg", "Rev_Type"]].corr(), annot = True)
plt.show()
