# -*- coding: utf-8 -*-
"""
Created on Thu Sep 27 23:11:03 2018

@author: archit bansal
"""

#import libraries and dataset from scikitlearn

import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
digits=load_digits()

#analyze a sample image
import pylab as pl
pl.gray()
pl.matshow(digits.images[0])
pl.show()

digits.images[0]

#visualize first 15 images
images_and_labels=list(zip(digits.images,digits.target))
plt.figure(figsize=(5,5))
for index,(image,label) in enumerate(images_and_labels[:15]):
    plt.subplot(3,5,index+1)
    plt.axis('off')
    plt.imshow(image,cmap=plt.cm.gray_r,interpolation='nearest')
    plt.title('%i'% label)
    
import random
from sklearn import ensemble

#define variables 
n_samples=len(digits.images)
x=digits.images.reshape((n_samples,-1))
y=digits.target

#splitting dataset into train and test set
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=0)


#using random forest classifier
classifier=ensemble.RandomForestClassifier()

#Fit model with sample data
classifier.fit(X_train,y_train)

#Attempt to predict validation data
score=classifier.score(X_test,y_test)
print(score)
y_pred=classifier.predict(X_test)


i=9
pl.gray() 
pl.matshow(digits.images[i]) 
pl.show() 
print(classifier.predict(x[i].reshape(1,-1)))