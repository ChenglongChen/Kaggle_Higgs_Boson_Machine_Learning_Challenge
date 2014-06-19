Kaggle_Higgs_Boson_Machine_Learning_Challenge
=============================================

This repo holds an example of R's GBM model for Higgs Boson Machine Learning Challenge. With the fixed params in the code, it scores 3.35741 on the public leaderboard and beats the MultiBoost benchmark (3.34085) by a bit. However, there are still rooms for improvement with further tuning the params. I once got to ~3.5x with this model but turned to XGboost right after the 3.6x benchmark code released since I found it hard to break 3.6x with R's GBM. If you happen to break that 'limit' with some modifications of my code, I would appreciate if you could drop a line here and let me know.

It implements a two-fold CV strategy based on the code I wrote for Kaggle's Loan Default Prediction Competition:
https://github.com/ChenglongChen/Kaggle_Loan_Default_Prediction
[It is actually a revised version with fixes for some leakages in previous CV code]

In specific, the inner CV is used to choose the best cutoff to maximize the AMS, while the outer CV is for estimating the performance using the chosen best cutoff. If you want to save time, just specify the best cutoff as, e.g.,0.85, and then run makeSubmission function. If however you want to have a sense of how your model is doing when you are tuning the params, I would suggest you to turn both inner CV and outer CV on.

As a last note, such two-fold CV strategy can be used for other similar situation.
