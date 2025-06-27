# 📱 SMS Spam Classifier

An end-to-end NLP pipeline that automatically flags spam in SMS messages using classical machine-learning models.

---

## 🔍 Overview

1. **Data**  
   - **Source:** SMS Spam Collection v1  
   - **Format:** `label\tmessage` (ham/spam)

2. **Preprocessing**  
   - Strip URLs, emails, numbers, punctuation  
   - Expand contractions, lowercase  
   - Tokenize, remove stop-words, lemmatize  

3. **Feature Extraction**  
   - TF-IDF vectors (unigrams & bigrams)  

4. **Modeling & Tuning**  
   - Train/test split (80/20)  
   - Compare 7 classifiers:  
     - Multinomial Naïve Bayes  
     - Logistic Regression  
     - Decision Tree  
     - Random Forest  
     - Gradient Boosting  
     - Support Vector Machine  
     - k-Nearest Neighbors  
   - Select best by accuracy  

5. **Evaluation**  
   - Confusion matrix  
   - Accuracy summary  

6. **Usage**  
   ```python
   import joblib
   model = joblib.load("models/best_model.pkl")
   print(model.predict([
       "Free entry in 2 a wkly comp to win FA Cup finals!", 
       "Are we still meeting at 6pm?"
   ]))
   # ['spam', 'ham']
