Kaggle_Higgs_Boson_Machine_Learning_Challenge
=============================================

#R's GBM model for Higgs Boson Machine Learning Challenge

This repo holds an example of R's GBM model for Higgs Boson Machine Learning Challenge. It implements a two-fold CV strategy based on the code I wrote for Kaggle's Loan Default Prediction Competition. In specific, the inner CV is used to choose the best cutoff to maximizing the AMS, while the outer CV is for estimating the performance using the chosen best cutoff. If you want to save time, just specify the best cutoff, e.g., as 0.85, and then run makeSubmission. If however you want to have a sense of how your model doing when you are tuning the params, I would suggest turn both inner CV and outer CV on.

As a last note, such two-fold CV strategy can be used for other similar situation.
