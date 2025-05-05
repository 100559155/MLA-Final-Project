### MLA Final Project 
### April Porter (100554572), Lucia Enriquez (100559155), and Filimon Keleta (100559040)

## Task 1: Text Preprocessing and Vectorization 
We now explore the use of Natural Language Processing (NLP) techniques to understand patterns in user reviews of over 5,000 movies. The goal was to investigate whether textual sentiment and content could reliably explain or predict the numeric movie ratings. Several vectorization strategies were used to convert review texts into numerical form, followed by a correlation analysis to understand the strength of their relationships with the rating. The analysis takes a comprehensive approach: we preprocess the reviews, test different text vectorization strategies (e.g., BoW, TF-IDF, Word2Vec, LDA), and evaluate their effectiveness in terms of correlation with the user-assigned movie ratings.
"The textual sentiment and structure of a movie review contain enough information to reflect or predict the reviewer's numeric rating."
We posited that:
- Reviews with more positive sentiment would correspond to higher ratings.
- Thematic patterns (captured by topic modeling) might differentiate high-rated movies from low-rated ones.
- Context-aware vectorizations (e.g., Word2Vec, TF-IDF with sentiment) would provide more accurate representations than basic methods like BoW.

# Preprocessing Pipeline
Before any vectorization, the reviews underwent a cleaning and normalization pipeline using spaCy. The following steps were applied:
- Lowercasing: Standardizes the text for consistent vectorization.
- Punctuation and Stop Word Removal: Filters out common, less informative words.
- Lemmatization: Reduces words to their root form (e.g., "watching" becomes "watch").
- Alphabet Filtering: Retains only alphabetic tokens, removing digits and special characters.
This step was essential to eliminate noise and ensure high-quality input for vector models. Sample output of preprocessing included tokens like ["film", "beautiful", "performance"], which are semantically strong and focused.

# Vectorization Strategies
After preprocessing, we applied the following strategies to convert the cleaned text into numerical vectors:
**Bag of Words (BoW)**
- Technique: Counted word frequencies across documents.
- Why: Simple baseline, interpretable.
- Result: Correlation with rating = 0.1967.
- Pros: Easy to implement, good baseline.
- Cons: Ignore context and semantic meaning.

**TF-IDF (with and without sentiment integration)**
- Technique: Weighted rare but meaningful words higher.
- Sentiment extension: Integrated SentiWordNet scores to capture emotional polarity.
- Result:
  - Raw TF-IDF correlation = 0.2309
  - Sentiment-weighted TF-IDF correlation = 0.2539 (best performer)
- Pros: Strong predictive quality, emphasizes unique, sentiment-laden words.
- Cons: Still bag-of-words-based; lacks deep semantic context.

**Word2Vec (spaCy and Google’s model (billions of trained tokens))**
- In this section to be positively evaluated, we also used a Google trained token called “GoogleNews-vectors-negative300”  to provide further analysis into words and sentiment. 
- Technique: Averaged word embeddings to create document vectors.
"Google News embeddings are trained on formal news text, while movie reviews often contain slang, informal language, or sarcasm."
- Result:
  - SpaCy embeddings: -0.0254
  - Google model embeddings: -0.1180
- Pros: Encodes word meaning.
- Cons: Averaging diluted specific meanings; poor correlation due to context loss.

**Topic Modeling (LDA)**
- Technique: Trained Latent Dirichlet Allocation with 15 topics and 40 passes.
- Result:
  - Coherence Score: 0.4040
  - Top topic correlation with rating: 0.0590
- Pros: Great for interpreting thematic structures.
- Cons: Weak correlation to numeric ratings; topics may not align with sentiment.

**TextBlob Sentiment Scores**
- Technique: Polarity scores from TextBlob.
- Result: Correlation = 0.2459
- Pros: Quick and intuitive.
- Cons: Oversimplifies nuanced sentiment.

# Results and Analysis
![Screenshot 2025-05-05 at 2 21 36 PM](https://github.com/user-attachments/assets/e3f7cb98-f822-45f9-89d1-17b33ee2db48)

The Sentiment-aware TF-IDF had the highest correlation with rating, validating our hypothesis that sentiment is a key factor. Surprisingly, Word2Vec models underperformed, likely due to domain mismatch and the information loss from averaging word vectors. LDA showed potential for interpretability but lacked predictive power.

# Recommendation
Based on these findings, the following actions are recommended:
1. Continue using TF-IDF with sentiment integration for tasks involving review classification or rating prediction.
2. Avoid averaging Word2Vec for document representation. Consider using Doc2Vec or transformer-based models (e.g., BERT) for richer embeddings.
3. Enhance topic modeling with bigram/trigram tokenization and guided LDA using seed words.
4. For production models, integrate ensemble approaches combining TF-IDF, sentiment, and topic distributions as features.
Conclusion
This analysis confirmed that textual sentiment correlates well with user ratings. Traditional models like TF-IDF (when enhanced with sentiment) outperform more complex, unsupervised embeddings like Word2Vec in this task. Preprocessing played a critical role in cleaning the data, and future work should focus on supervised learning techniques and model ensembles to maximize prediction accuracy and interpretability. This demonstrates that even in large, noisy text datasets, relatively simple strategies—if thoughtfully executed—can yield meaningful insights.

## Task 2: Machine Learning Model 
	The aim of our project was to use the data we had generated (the names of movies, their averages ratings, and their reviews) to predict what the rating of the movie would be. In Task 1, we used BoW, TF-IDF, Word2Vec, Google Word2Vec, LDA Topic Modeling, and TextBlob to create vectorizations for the reviews and allow us to use the data for regression. To create the ML model for this project, we used Select K-Best and Linear Regression on each vectorization scheme to determine what would have the strongest correlation. Linear Regression turned out to have fairly weak results so we switched to a Random Forest model, which we used for the final results of our project.   
Select K-Best used the top 100 topics for each movie review and analyzed the root mean-squared error (RMSE) using this training data. We saw low correlation between predicted ratings and actual ratings for each movie using this model and decided to run the same data through a Random Forest model to compare its predictive abilities. 
The Random Forest model uses multiple decision trees to improve the accuracy of each predictive data point. The model averages each prediction for every movie rating from the different decision trees. The group used the SCIKIT learn package in python which provides tools for Random Forest modeling. The RMSE is a calculation taken from subtracting the predicted value from the true value and then dividing the result by the number of data points. To normalize the data, RMSE is the square root of this calculation.

# Model Creation

# Comparison of Models
The below data is also available in our GitHub repository under the name “mla_train_analysis.csv.”
![Screenshot 2025-05-05 at 2 22 02 PM](https://github.com/user-attachments/assets/698a06d9-f414-4594-b88d-120ecbab9867)

	RMSE is in the range 0 to infinity where numbers closest to 0 mean stronger correlation and R-Squared is in the range negative infinity to 1. If R-Squared equals 1 then the predictions are perfect, if it equals 0 then the model does no better than predicting the mean of the target, and if it is less than 0 then the model is worse than using the mean predictor. RMSE is more sensitive to outliers because it squares the errors. 
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
The dashboard task was realized by using the python library Dash, which has built-in visualization tools for our final graphs which were 1 histogram and 2 scatterplots. The group imported the trained MLA models for each vectorization technique using .pkl files, which preserve the predictive abilities of each type of model using its given vectorization technique and the imported movie reviews which it was trained on. The .pkl files used were with the BoW, LDA, and Textblob Sentiment vectorization techniques. 


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

