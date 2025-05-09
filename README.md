### MLA Final Project 
### April Porter (100554572), Lucia Enriquez (100559155), and Filimon Keleta (100559040)

## Task 1: Text Preprocessing and Vectorization 
	- The file "movie_scraper.py" is the code we used to generate the dataset.
	- The file "5kmovies.csv" includes the name, ratings, and reviews of every movie we used. 

We now explore the use of Natural Language Processing (NLP) techniques to understand patterns in user reviews of over 5,000 movies. The goal was to investigate whether textual sentiment and content could reliably explain or predict the numeric movie ratings. Several vectorization strategies were used to convert review texts into numerical form, followed by a correlation analysis to understand the strength of their relationships with the rating. The analysis takes a comprehensive approach: we preprocess the reviews, test different text vectorization strategies (e.g., BoW, TF-IDF, Word2Vec, LDA), and evaluate their effectiveness in terms of correlation with the user-assigned movie ratings.
"The textual sentiment and structure of a movie review contain enough information to reflect or predict the reviewer's numeric rating."
We posited that:
- Reviews with more positive sentiment would correspond to higher ratings.
- Thematic patterns (captured by topic modeling) might differentiate high-rated movies from low-rated ones.
- Context-aware vectorizations (e.g., Word2Vec, TF-IDF with sentiment) would provide more accurate representations than basic methods like BoW.

# Preprocessing Pipeline
	- The file "5k_movies_preprocssed.csv" includes the movie data after pre-processing.
	- The file "preprocessor.py" has the code we used to prep-process the movie reviews.
	- The file "movie_nlp_analysis.csv" has the resulting data from the tokenization process.

Before any vectorization, the reviews underwent a cleaning and normalization pipeline using spaCy. The following steps were applied:
- Lowercasing: Standardizes the text for consistent vectorization.
- Punctuation and Stop Word Removal: Filters out common, less informative words.
- Lemmatization: Reduces words to their root form (e.g., "watching" becomes "watch").
- Alphabet Filtering: Retains only alphabetic tokens, removing digits and special characters.
This step was essential to eliminate noise and ensure high-quality input for vector models. Sample output of preprocessing included tokens like ["film", "beautiful", "performance"], which are semantically strong and focused.

# Vectorization Strategies
	- The file "vectorize.py" has the code we used to vectorize the movie reviews.

After preprocessing, we applied the following strategies to convert the cleaned text into numerical vectors:
**Bag of Words (BoW)**
	- The file "BoW Avg_model.pkl" has the ML model generated for this section.

- Technique: Counted word frequencies across documents.
- Why: Simple baseline, interpretable.
- Result: Correlation with rating = 0.1967.
- Pros: Easy to implement, good baseline.
- Cons: Ignore context and semantic meaning.

**TF-IDF (with and without sentiment integration)**
	
 	- The file "TF-IDF Avg_model.pkl" has the ML model generated for this section.

- Technique: Weighted rare but meaningful words higher.
- Sentiment extension: Integrated SentiWordNet scores to capture emotional polarity.
- Result:
  - Raw TF-IDF correlation = 0.2309
  - Sentiment-weighted TF-IDF correlation = 0.2539 (best performer)
- Pros: Strong predictive quality, emphasizes unique, sentiment-laden words.
- Cons: Still bag-of-words-based; lacks deep semantic context.

**Word2Vec (spaCy and Google’s model (billions of trained tokens))**
	
 	- The file "Word2Vec Avg_model.pkl" has the ML model generated for this section.
	- The file "Google Word2Vec Avg_model.pkl" has the ML model generated for this section.

- In this section to be positively evaluated, we also used a Google trained token called “GoogleNews-vectors-negative300”  to provide further analysis into words and sentiment. 
- Technique: Averaged word embeddings to create document vectors.
"Google News embeddings are trained on formal news text, while movie reviews often contain slang, informal language, or sarcasm."
- Result:
  - SpaCy embeddings: -0.0254
  - Google model embeddings: -0.1180
- Pros: Encodes word meaning.
- Cons: Averaging diluted specific meanings; poor correlation due to context loss.

**Topic Modeling (LDA)**
	
 	- The file "LDA_Topic_Model.pkl" has the ML model generated for this section.

- Technique: Trained Latent Dirichlet Allocation with 15 topics and 40 passes.
- Result:
  - Coherence Score: 0.4040
  - Top topic correlation with rating: 0.0590
- Pros: Great for interpreting thematic structures.
- Cons: Weak correlation to numeric ratings; topics may not align with sentiment.

**TextBlob Sentiment Scores**
	
 	- The file "TextBlob Sentiment_model.pkl" has the ML model generated for this section.

- Technique: Polarity scores from TextBlob.
- Result: Correlation = 0.2459
- Pros: Quick and intuitive.
- Cons: Oversimplifies nuanced sentiment.

# Results and Analysis
![Screenshot 2025-05-05 at 2 21 36 PM](https://github.com/user-attachments/assets/e3f7cb98-f822-45f9-89d1-17b33ee2db48)

The Sentiment-aware TF-IDF had the highest correlation with rating, validating our hypothesis that sentiment is a key factor. Surprisingly, Word2Vec models underperformed, likely due to domain mismatch and the information loss from averaging word vectors. LDA showed potential for interpretability but lacked predictive power.

**Model performance as a Visual Representation**
<img width="991" alt="Screenshot 2025-05-08 at 12 46 22 PM" src="https://github.com/user-attachments/assets/340ef204-d3a8-41ed-a7eb-2bff8a4ad6bb" />
<img width="988" alt="Screenshot 2025-05-08 at 12 45 34 PM" src="https://github.com/user-attachments/assets/70e2b68f-317f-4558-8926-cc75405f3269" />
<img width="988" alt="Screenshot 2025-05-08 at 12 45 49 PM" src="https://github.com/user-attachments/assets/1552c137-14b9-4340-bcac-0a2e0b2da3b2" />

# Recommendation
Based on these findings, the following actions are recommended:
1. Continue using TF-IDF with sentiment integration for tasks involving review classification or rating prediction.
2. Avoid averaging Word2Vec for document representation. Consider using Doc2Vec or transformer-based models (e.g., BERT) for richer embeddings.
3. Enhance topic modeling with bigram/trigram tokenization and guided LDA using seed words.
4. For production models, integrate ensemble approaches combining TF-IDF, sentiment, and topic distributions as features.
Conclusion
This analysis confirmed that textual sentiment correlates well with user ratings. Traditional models like TF-IDF (when enhanced with sentiment) outperform more complex, unsupervised embeddings like Word2Vec in this task. Preprocessing played a critical role in cleaning the data, and future work should focus on supervised learning techniques and model ensembles to maximize prediction accuracy and interpretability. This demonstrates that even in large, noisy text datasets, relatively simple strategies—if thoughtfully executed—can yield meaningful insights.

## Task 2: Machine Learning Model 
	- The files "ML_Train.py" and "ML_Train_RandomForest.py" have the code we used for this section.
	- The files "mla_train_analysis.csv" and "mla_train_analysis_RandomForest.csv" have the resulting data.

The aim of our project was to use the data we had generated (the names of movies, their averages ratings, and their reviews) to predict what the rating of the movie would be. In Task 1, we used BoW, TF-IDF, Word2Vec, Google Word2Vec, LDA Topic Modeling, and TextBlob to create vectorizations for the reviews and allow us to use the data for regression. To create the ML model for this project, we used Select K-Best and Linear Regression on each vectorization scheme to determine what would have the strongest correlation. Linear Regression turned out to have fairly weak results so we switched to a Random Forest + Ridge Regression model, which we used for the final results of our project.   
Select K-Best used the top 100 topics for each movie review and analyzed the root mean-squared error (RMSE) using this training data. We saw low correlation between predicted ratings and actual ratings for each movie using this model and decided to run the same data through a Ridge Regression model to compare its predictive abilities. 
The Ridge Regression Forest model uses multiple decision trees to improve the accuracy of each predictive data point. The model averages each prediction for every movie rating from the different decision trees. The group used the SCIKIT learn package in python which provides tools for Ridge Regression modeling. The RMSE is a calculation taken from subtracting the predicted value from the true value and then dividing the result by the number of data points. To normalize the data, RMSE is the square root of this calculation.

# Model Creation
The Ridge Regression model with hyperparameter tuning to predict the movie ratings based on various feature representations of the reviews. It splits the input feature X and labels y into training and testing sets (80/20 splits). A pipeline was defined with three steps, StandardScaler (normalizes the features), SelectKBest (selects the best features using f_regression, though k=’all’ means it uses all features), and Ridge (applies Ridge Regression). We also did Hyperparameter Tuning which uses GridSearchCV to test different values of the regularization parameter alpha for Ridge Regression ([0.01, 0.1, 1.0, 10.0, 100.0]) with 5 -fold cross-validation. To evaluate the model we predict ratings on the test set. 

This calculates and prints the:
- RMSE (Root Mean Squared Error)
- R2 Score (Coefficient of determination)

It eventually returns RMSE, R2 Score and the best model from GridSearchCV.

We run the MLA Model on each feature set (Bag of Words, TF-IDF, Word2Vec, Sentiment Scores, and LDA with 15 Topics). For LDA, we extract the topic distribution for each review, and use this topic matrix as input to mla_model for regression analysis. This automates the process of evaluating multiple text-based feature extraction techniques using Ridge Regression, optimizing the model via cross-validation, and saving both performance results and the trained models for further use.

# Comparison of Models
The below data is also available in our GitHub repository under the name “mla_train_analysis.csv.”
![Screenshot 2025-05-05 at 2 22 02 PM](https://github.com/user-attachments/assets/698a06d9-f414-4594-b88d-120ecbab9867)

RMSE is in the range 0 to infinity where numbers closest to 0 mean stronger correlation and R-Squared is in the range negative infinity to 1. If R-Squared equals 1 then the predictions are perfect, if it equals 0 then the model does no better than predicting the mean of the target, and if it is less than 0 then the model is worse than using the mean predictor.  
# Analysis of Results 
**BoW** 
~4% of variance explained, and relatively high error.
	**TF-IDF**
Slightly better than BoW but generally similar results.
**Word2Vec**
This result is worse than the mean predictor. The RMSE is higher than previous examples and the R² is negative, meaning the model performs worse than predicting the average of the target variable.
**Google Word2Vec**
Slightly worse than Word2Vec but generally similar results.
**TextBlob**
Better results than BoW and TF-IDF but still only explains 5.6% of the variance in the model.
**LDA Topic Distribution**
This model had the best performance out of the group (lowest RMSE and highest R-Squared), explaining approximately 13% of the variance.

As we can see from the table and the analysis, LDA Topic Distribution was the strongest method for the Random Forest model. We believe LDA performed best because of the way it represents the text itself. LDA captures documents as distributions over topics which provide a lower-dimensional and more meaningful summary of the review text. This means that the results are likely to align better with the target variable and help with creating more nuanced predictions. LDA is also best with longer texts and therefore may have been less effective for movies with few reviews. 
One thing we noted in our dataset is that the majority of movies have an average rating of about 6 or 7, and there are very few movies rated very high or very low. For this reason, the model appeared to struggle to predict the ratings for movies that were outliers. We used approximately 4,700 data points, which was the result of pulling the reviews for 10,000 randomly selected movies and removing the ones that had no reviews. Because there are only so many movies that have reviews (must be recent enough to be posted online and famous enough to receive reviews), we were limited by the amount and breadth of data that was available. 


## Task 3: Implementation of a dashboard
	- The file "dashboard_.py" is our implementation of the dashboard.
 
The dashboard task was realized by using the python library Dash, which has built-in visualization tools for our final graphs which were 1 histogram and 2 scatterplots. The group imported the trained MLA models for each vectorization technique using .pkl files, which preserve the predictive abilities of each type of model using its given vectorization technique and the imported movie reviews which it was trained on. The .pkl files used were with the BoW, LDA, and Textblob Sentiment vectorization techniques. The dashboard displays 3 graphs, as mentioned previously: the first graph displayed at the top of the user’s browser displays the predicted rating based on each vectorization technique: BoW, TF-IDF, LDA, and Word2Vec. Thus, the user can see in real time how their word usage is being inputted to the machine learning model and analyzed into the form of a predicted rating. Directly below this graph is another one that displays the historical rating data of the user based on how many times they input a value into the review textbox, shown on the top-left. 
In addition, the dashboard takes in a review from a user on a pretrained model and uses that to analyze sentiment scores, which is displayed on the top-right side of the user’s screen. This is based on word polarity where words are either classified negative, positive or neutral. Movie selection is based on available movies (movies used for training and vectorizing models), this prompts the dash to display current average rating and review count. After the user puts in a review, it automates preprocessing (lemmatization, stopword removal) in addition to sentiment analysis. Total prediction based on 1/RMSE of each model weighed across each model for scaled average. 

The visual analytics includes:
1. Rating comparison: *Bar chart showing all model predictions* 
2. Sentiment Breakdown: *Pie chart of Sentiment distribution* 
3. Historical Trends: *Line graph of past ratings*
4. Word Cloud: *Visual representation of Key review terms*
5. Model Performance: *Scatter plots with RMSE/R2 Metrics*


## Section 4: Acknowledgement of Authorship 

**Task 1 References:**

*Text vectorization examples*

SpaCy Tutorial from class, Author: Jerónimo Arenas-García, Date: Feb, 2024

**Task 2 References:**

*K-Best regression model set up and pipeline execution code*

Rawanreda. “Feature Selection Techniques Tutorial.” Kaggle, Kaggle, 22 June 2020, www.kaggle.com/code/rawanreda/feature-selection-techniques-tutorial.  
“Selectkbest.” Scikit, [scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectKBest.html](url).  Accessed 5 May 2025. 

*Adding BoW and TF-IDF vectorization models into training code*

Bisman. “Logistic Regression - Bow and TFIDF.” Kaggle, Kaggle, 8 July 2019, [www.kaggle.com/code/bisman/logistic-regression-bow-and-tfidf](url).  

**Task 3 References:** 

*Set up code for implementing the python dashboard*

Castillo, Dylan. Develop Data Visualization Interfaces in Python With Dash. 2 Feb. 2025, [realpython.com/python-dash](url). 

