# Traffic accidents Analysis and Prediction

### Objective:
The primary goal of this project was to analyze traffic accidents data and develop a predictive model capable of forecasting the likelihood of accidents under various conditions. The analysis aimed to identify key factors contributing to accidents and provide actionable insights for improving road safety.

### Dataset:
The project utilized a comprehensive dataset containing historical traffic accidents records. Key features included weather conditions, time of day, road type, vehicle type, driver behavior, and accident severity. The dataset was preprocessed to handle missing values, outliers, and categorical variables, ensuring high-quality input for the machine learning models.

### Methodology:
*Data Preprocessing*:
   - Cleaned and normalized the dataset.
   - Encoded categorical variables using techniques like one-hot encoding.
   - Split the data into training and testing sets (80% training, 20% testing).

*Exploratory Data Analysis (EDA)*:
   - Performed statistical analysis and visualizations to understand feature distributions and correlations.
   - Identified significant patterns, such as higher accident rates during adverse weather conditions or peak traffic hours.

*Feature Selection*:
   - Selected the most relevant features using techniques like correlation analysis and feature importance ranking.

*Model Development*:
   - Experimented with multiple machine learning algorithms, including Logistic Regression, Decision Trees, Support Vector Machines (SVM), and Random Forest Classifier.
   - Tuned hyperparameters using Grid Search and Cross-Validation to optimize model performance.

*Evaluation Metrics*:
   - Evaluated models based on accuracy, precision, recall, F1-score, and ROC-AUC.
   - The Random Forest Classifier achieved the highest accuracy of 82%, demonstrating its effectiveness in predicting traffic accidents.

### Key Findings:
- Weather conditions (e.g., rain, snow) and time of day were among the most influential factors contributing to accidents.
- Certain road types and driver behaviors showed strong correlations with accident severity.
- The Random Forest Classifier outperformed other algorithms due to its ability to handle complex interactions between features and reduce overfitting.

### Tools and Technologies:
- *Programming Language*: Python
- *Libraries*: Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn
- *Algorithm*: Random Forest Classifier
- *Accuracy Achieved*: 82%
- *Evaluation Metrics*: Accuracy, ROC-AUC, Precision, Recall

### Analysis
*Hour with the most accidents*: 17:00 (16097 accidents)  
*Hour with the least accidents*: 4:00 (2104 accidents)  
*Month with the most accidents*: 10 (20089 accidents)  
*Month with the least accidents*: 2 (14621 accidents)  
*Day with the most accidents*: Sunday (34458 accidents)  
*Day with the least accidents*: Tuesday (25246 accidents)


####  Injury Report: Total Injured and Not Injured
Injured : 117376  
Not Injured : 91930

### Output
|Model|Accuracy score|
|-----|--------------|
|LogisticRegression|0.8277196502794898|
|RandomForestClassifier|0.8328077970474416|
|DecisionTreeClassifier|0.787898332616693|


|Model|MSE|MAE|R2|
|-----|---|---|--|
|Logistic_Regression|0.172280|0.172280|0.300079|
|Random_Forest|0.168243|0.168243|0.316480|
|Decison_Tree_Classifier|0.212149|0.212149|0.138103|

## Conclusion

In this project, we analyzed traffic accidents data to predict whether an accident would result in injury or not. Using the Random Forest Classifier algorithm, we achieved an 83% accuracy in predicting the outcome of traffic accidents. This model was trained on various features like weather conditions, road types, time of day, and vehicle types to identify patterns that contribute to injuries.
