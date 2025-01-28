## Literature Review: Credit Card Fraud Detection 
# Background and Motivation
Credit card fraud is one of the most prevalent financial crimes, causing substantial economic losses and eroding consumer trust in financial institutions. With the increasing volume of transactions, traditional methods of fraud detection, such as supervised learning, often struggle due to the imbalanced nature of fraud data—fraudulent transactions are rare, making it difficult for models to learn meaningful patterns from labeled data. This problem is exacerbated by the fact that fraud tactics evolve rapidly, making it hard for static models to keep up. Therefore, there is a growing interest in anomaly detection techniques, which are better suited to identifying rare events without needing labeled data for every possible fraud scenario.
The paper under review explores the potential of anomaly detection methods in fraud detection, specifically focusing on techniques such as Isolation Forest, One-Class Support Vector Machine (SVM), Local Outlier Factor (LOF), and DBSCAN (Density-Based Spatial Clustering of Applications with Noise). These methods do not rely on labeled data for fraud detection and instead learn the patterns of normal transactions. Anything deviating significantly from this "normal" behavior is flagged as an anomaly. Given the rarity of fraud in financial data, anomaly detection offers a promising alternative to traditional supervised models, which often fail to generalize well on unseen fraudulent activities.
The motivation for this study stems from the need for scalable and effective fraud detection methods that are adaptable to new fraud tactics without requiring extensive retraining on labeled fraud data. The research also addresses the importance of feature selection, transformation, and the evaluation of anomaly detection algorithms to find the most suitable models for identifying fraudulent transactions in credit card datasets.

# Methods Used

Data Preprocessing and Feature Selection

The authors analyzed a dataset of credit card transactions, including anonymized features and transaction amounts. They addressed issues like skewed distributions, particularly with transaction amounts, by applying log transformations to make the data more suitable for modeling.

Hypothesis Testing: Z-Test for Feature Significance

To refine feature selection, the authors performed a Z-test to check if the differences between valid and fraudulent transactions were statistically significant. This helped identify the most important features for detecting fraud.

Anomaly Detection Algorithms

Several unsupervised algorithms were used:

Isolation Forest: This technique isolates anomalies by randomly partitioning data, making it efficient for large datasets.

One-Class SVM: It learns the boundary of normal data and flags anything outside as an anomaly, effective for high-dimensional data.

![image](https://github.com/user-attachments/assets/def6c297-98e1-4396-b77e-8c14943158c9)


Local Outlier Factor (LOF): LOF detects anomalies based on local data density, ideal for identifying sparse fraud cases.

![image](https://github.com/user-attachments/assets/71ab796e-85cb-4a33-894f-9c9ea929f311)


DBSCAN: A clustering algorithm that identifies outliers in complex data by labeling points outside any clusters as potential fraud.

H0: There is no difference (insignificant)
H1: There is a difference (significant)

Formula for z-score
Zscore= (¯x−μ) / S.E

These methods were selected to handle the rare and complex nature of fraud in the dataset.
The models were evaluated using recall (to catch more fraud), precision (to reduce false positives), and cost analysis. While high recall is key, it often leads to false positives, which can frustrate customers and cause inefficiencies. The authors highlighted the need to balance recall and precision to minimize disruption while still detecting fraud effectively.

# Significance of the Work
This paper highlights the value of anomaly detection methods as an alternative to traditional supervised learning for fraud detection. Unlike supervised models, which need large amounts of labeled data, anomaly detection is well-suited for identifying rare events like fraud. Key findings include:

Feature Transformation: Applying log transformations to skewed features, such as transaction amounts, improved model performance by stabilizing variance and making the data more symmetric.

Recall vs. Precision: While DBSCAN achieved high recall (92%), it flagged many legitimate transactions as fraud. This highlights the typical trade-off between recall and precision in fraud detection.

Effective Models: One-Class SVM and Isolation Forest provided a better balance, catching around 87% of fraud while reducing false positives, making them more practical for real-world use.

These results show that anomaly detection can be an effective fraud detection tool, especially when labeled fraud data is limited. The paper underscores the importance of balancing recall and precision to avoid unnecessary disruptions for customers.

# Connection to Other Work
This study extends previous research in credit card fraud detection by exploring anomaly detection methods, which differ from traditional supervised techniques like logistic regression and random forests. While supervised models need large amounts of labeled fraud data, they often struggle due to the class imbalance—fraud is rare, so these models can fail to accurately detect it.

Anomaly detection methods, like Isolation Forest, One-Class SVM, and LOF, don’t rely on labeled fraud data and instead learn what "normal" transactions look like, making them ideal for fraud detection in data-poor environments. The study also aligns with earlier research suggesting that combining different anomaly detection techniques could improve detection accuracy in real-world scenarios.

# Relevance to Capstone Project
The findings from this paper are highly relevant for a capstone project focused on fraud or anomaly detection in financial transactions. Here’s how the key takeaways can apply:

Feature Engineering & Data Transformation: The importance of transforming skewed data (e.g., using log transformations) is something that can directly improve model performance in any fraud detection system.

Anomaly Detection Techniques: The paper’s review of algorithms like Isolation Forest, One-Class SVM, LOF, and DBSCAN offers valuable methods to explore in a capstone project, especially for identifying rare events like fraud.

Balancing Precision and Recall: The trade-off between recall and precision is a critical issue in fraud detection. A capstone project could dive deeper into optimizing this balance, perhaps through hybrid models or post-processing steps to minimize false positives.

For a project, you could build on this research by adding more features, experimenting with hybrid approaches combining supervised and unsupervised methods, or even working on a real-time fraud detection system. Additionally, testing these models on larger, diverse datasets could provide further insights into performance.
![image](https://github.com/user-attachments/assets/7fc062a0-4630-4767-bfa1-32525fa48477)


# Conclusion

The study shows how anomaly detection techniques can be highly effective for credit card fraud detection, especially when dealing with rare and evolving fraud patterns. Key insights, like the importance of selecting the right features and balancing recall with precision, are essential for building effective fraud detection systems. The research also suggests that combining different anomaly detection methods can provide more reliable and practical solutions. Overall, this work adds valuable knowledge to the field of anomaly detection and offers useful directions for future research in fraud detection and financial security.

# References
https://www.kaggle.com/code/abdocan/credit-card-fraud-detection-if-lof-dbscan

https://www.stepbystepdatascience.com/fraud-detection-with-dbscan-and-tsne






