import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

wine_df = pd.read_csv(r'C:\Users\User\PycharmProjects\winter2020-code\6-python-ml-notebook\data\wine.data',
                      sep=',',
                      header=None)
wine_df.columns = ['class', 'alcohol', 'malic_acid', 'ash', 'alcalinity_of_ash', 'magnesium', 'total_phenols',
                   'flavanoids',
                   'nonflavanoid_phenols', 'proanthocyanins', 'color_intensity', 'hue', 'od280 od315_of_diluted_wines',
                   'proline']

wine_df= wine_df.drop(['od280 od315_of_diluted_wines','nonflavanoid_phenols', 'ash','magnesium','alcalinity_of_ash'], axis=1)
os.makedirs('Charts', exist_ok=True)

legen = ['Class 1','Class 2','Class 3']

filter1 = wine_df['class'] == 1
filter2 = wine_df['class'] == 2
filter3 = wine_df['class'] == 3

C1 = wine_df.loc[filter1,['hue']].values
C2 = wine_df.loc[filter2,['hue']].values
C3 = wine_df.loc[filter3,['hue']].values

fig, ax = plt.subplots(1, 1, figsize=(5, 5))
ax.hist([C1,C2,C3], bins=16,  histtype='bar', stacked=True, color=['green','blue','red'], edgecolor='Black', alpha=0.81)
ax.set_title('Histogram of Hue by class')
ax.legend(legen,loc=2)
plt.savefig('Charts/hue.png', dpi=300)


C1 = wine_df.loc[filter1,['alcohol']].values
C2 = wine_df.loc[filter2,['alcohol']].values
C3 = wine_df.loc[filter3,['alcohol']].values

fig, ax = plt.subplots(1, 1, figsize=(5, 5))
ax.hist([C1,C2,C3], bins=16,  histtype='bar', stacked=True, color=['green','blue','red'], edgecolor='Black', alpha=0.81)
ax.set_title('Histogram of Hue by class')
ax.legend(legen,loc=2)
plt.savefig('Charts/alcohol.png', dpi=300)

C1 = wine_df.loc[filter1,['proline']].values
C2 = wine_df.loc[filter2,['proline']].values
C3 = wine_df.loc[filter3,['proline']].values

fig, ax = plt.subplots(1, 1, figsize=(5, 5))
ax.hist([C1,C2,C3], bins=16,  histtype='bar', stacked=True, color=['green','blue','red'], edgecolor='Black', alpha=0.81)
ax.set_title('Histogram of Hue by class')
ax.legend(legen)
plt.savefig('Charts/proline.png', dpi=300)


C1 = wine_df.loc[filter1,['flavanoids']].values
C2 = wine_df.loc[filter2,['flavanoids']].values
C3 = wine_df.loc[filter3,['flavanoids']].values

fig, ax = plt.subplots(1, 1, figsize=(5, 5))
ax.hist([C1,C2,C3], bins=16,  histtype='bar', stacked=True, color=['green','blue','red'], edgecolor='Black', alpha=0.81)
ax.set_title('Histogram of Hue by class')
ax.legend(legen)
plt.savefig('Charts/flavanoids.png', dpi=300)


C1 = wine_df.loc[filter1,['color_intensity']].values
C2 = wine_df.loc[filter2,['color_intensity']].values
C3 = wine_df.loc[filter3,['color_intensity']].values

fig, ax = plt.subplots(1, 1, figsize=(5, 5))
ax.hist([C1,C2,C3], bins=16,  histtype='bar', stacked=True, color=['green','blue','red'], edgecolor='Black', alpha=0.81)
ax.set_title('Histogram of Hue by class')
ax.legend(legen)
plt.savefig('Charts/color_intensity.png', dpi=300)


C1 = wine_df.loc[filter1,['total_phenols']].values
C2 = wine_df.loc[filter2,['total_phenols']].values
C3 = wine_df.loc[filter3,['total_phenols']].values

fig, ax = plt.subplots(1, 1, figsize=(5, 5))
ax.hist([C1,C2,C3], bins=16,  histtype='bar', stacked=True, color=['green','blue','red'], edgecolor='Black', alpha=0.81)
ax.set_title('Histogram of Hue by class')
ax.legend(legen)
plt.savefig('Charts/total_phenols.png', dpi=300)


C1 = wine_df.loc[filter1,['proanthocyanins']].values
C2 = wine_df.loc[filter2,['proanthocyanins']].values
C3 = wine_df.loc[filter3,['proanthocyanins']].values

fig, ax = plt.subplots(1, 1, figsize=(5, 5))
ax.hist([C1,C2,C3], bins=16,  histtype='bar', stacked=True, color=['green','blue','red'], edgecolor='Black', alpha=0.81)
ax.set_title('Histogram of Hue by class')
ax.legend(legen)
plt.savefig('Charts/proanthocyanins.png', dpi=300)


#Now we create a heatmap of correlations

fig, axes = plt.subplots(1, 1, figsize=(30, 30))
correlation = wine_df.corr().round(2)
im = axes.imshow(correlation, cmap='Blues')
cbar = axes.figure.colorbar(im, ax=axes)
cbar.ax.set_ylabel('Correlation', rotation=-90, va="bottom", fontsize=25)
numrows = len(correlation.iloc[0])
numcolumns = len(correlation.columns)
axes.set_xticks(np.arange(numrows))
axes.set_yticks(np.arange(numcolumns))
axes.set_xticklabels(correlation.columns, fontsize=25)
axes.set_yticklabels(correlation.columns, fontsize=25)
plt.setp(axes.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')
for i in range(numrows):
    for j in range(numcolumns):
        text = axes.text(j, i, correlation.iloc[i, j], ha='center', va='center', color='Black', fontsize=25)
axes.set_title('Heatmap of Correlation of Dimensions')
fig.tight_layout()
plt.savefig('Charts/heat.png', dpi=300)


#...
#...
#Machine learning part
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split

wine = load_wine()

feature_names = wine.feature_names
X = wine.data
y = wine.target

(X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size=0.35, random_state=1)

from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()
lr.fit(X_train, y_train)

# Output of the training is a model: a + b*X0 + c*X1 + d*X2 ...
print(f"Intercept per class: {lr.intercept_}\n")
print(f"Coeficients per class: {lr.coef_}\n")
print(f"Available classes: {lr.classes_}\n")
print(f"Named Coeficients for class 1: {pd.DataFrame(lr.coef_[0], feature_names)}\n")
print(f"Named Coeficients for class 2: {pd.DataFrame(lr.coef_[1], feature_names)}\n")
print(f"Named Coeficients for class 3: {pd.DataFrame(lr.coef_[2], feature_names)}\n")

print(f"Number of iterations generating model: {lr.n_iter_}")

# Predicting the results for our test dataset
predicted_values = lr.predict(X_test)

# Printing the residuals: difference between real and predicted
for (real, predicted) in list(zip(y_test, predicted_values)):
    print(f'Value: {real}, pred: {predicted} {"is different" if real != predicted else ""}')

# Printing accuracy score(mean accuracy) from 0 - 1
print(f'Accuracy score is {lr.score(X_test, y_test):.2f}/1 \n')

# Printing the classification report
from sklearn.metrics import classification_report, confusion_matrix, f1_score

print('Classification Report')
print(classification_report(y_test, predicted_values))

# Printing the classification confusion matrix (diagonal is true)
print('Confusion Matrix')
print(confusion_matrix(y_test, predicted_values))

print('Overall f1-score')
print(f1_score(y_test, predicted_values, average="macro"))