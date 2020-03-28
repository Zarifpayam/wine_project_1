# wine_project_1
This project tries to generate more insight about how wines are classified based on a few metrics that are measurable using precise tools. 
Some of the measures include Hue, Color intensity, Proline (which is a type of protein), Alcohol, which are pretty self explanatory. The dataset
we use is a Toy dataset courtasy of SKlearn package in python. It is composed, entirely, of numbers. There are no strings. However, there the class
of wine, unlike all other features in the dataset, is a categorical feature, its categories being 1,2 and 3.

We accomplish this project in two steps:

  - Step 1: The goal of this step is to reach a better understanding of our data.
            We explore each defining measure for wines with the help of a histogram. This way we can see how the values of that measure             are distributed. 
            The more spread out the histogram, the higher the chance that the measure in question will have a meaningful impact on                   others.
            We also add the dimension of class to each of the above histograms to see how wine classes fair against the distribution of             each measure. 
            The findings from this step are stored in the data folder in the repo. Here is an example: 
            

<img src="Charts/hue.png" width=400 height=400>
            
     
  - Step 2: The goal of this step is to predict the class to which a wine will belong, based on its defining measures. 
            To do so, we leverage the machine learning library SKlearn in python. The method we use is logistic regression (LR).
            We choose to keep all the measures or estimators for the LR since the output of the correlation matrix ensured us that there 
            is not a high correlation between any two pairs of measures. Thus, it was safe to say that the effect of multi-collinearity             would not produce: 
            
            
 ![](https://github.com/Zarifpayam/wine_project_1/blob/master/Charts/heat.png)
