<div id="top" align="center">
</div>
<!-- PROJECT LOGO -->
<div align="center">
<h1 align="center">Data Mining Project</h1>
  <h3 align="center">
    Applying nine classification algorithms for evaluating the efficiency on the Airline Passenger Satisfaction Dataset
    <br />
    <br />
    <a href="https://github.com/dngcphngnh04/DataMining-Project/issues">Report Bug</a>
    ·
    <a href="https://github.com/dngcphngnh04/DataMining-Project/issues">Request Feature</a>
  </h3>

[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Issues][issues-shield]][issues-url]

</div>

# ABOUT

## 1. The team behind it

| No. |       Full Name        | Student's ID |              Email               |                  Github account                   |                               Roles                                |
| :-: | :--------------------: | :----------: | :------------------------------: | :-----------------------------------------------: | :----------------------------------------------------------------: |
|  1  | Dương Ngọc Phương Anh | ITDSIU20090  | ITDSIU20090@student.hcmiu.edu.vn |     [dngcphngnh04](https://github.com/dngcphngnh04)     |   **TEAM LEADER** with Model evaluation, Hyper-parameters tuning   |
|  2  | Nguyễn Minh Đạt | ITDSIU20031  | ITDSIU20031@student.hcmiu.edu.vn |    [29Schiller](https://github.com/29Schiller)    |              Data preprocessing, Optimize performance              |
|  3  | Nguyễn Hải Phú | ITDSIU20100  | ITDSIU20100@student.hcmiu.edu.vn |     [haiphu241](https://github.com/haiphu241)     | Implement classification/prediction algorithms in Java, Bug fixing |


## 2. The project we are working on

The project investigates depression in Vietnamese teenagers aged 15-25 using modern data mining methods, highlighting
the need for culturally sensitive early detection strategies to reduce mental health issues.

## 3. Goal

- Use machine learning models to accurately predict depression levels among Vietnamese students.
- Tests a range of complex machine learning model approaches to identify the most successful ones.
- Increase the endurance and effectiveness of these classification models by including more potent algorithms.
- Apply data mining techniques to real-world datasets to bridge the gap between theoretical research and practical
  application.
- Empower healthcare practitioners to make more informed clinical decisions and enhance the mental health of Vietnamese
  students through early detection and intervention.

# REASON

## 1. Motivation

Depression awareness in Vietnam has increased due to the Internet, but the public's willingness to seek screenings
remains low. Self-assessment programs like "15minutes4me" offer hope, but they often rely on predetermined symptoms,
overlooking the complex interplay between living environment and stress. This data mining project aims to predict
depression levels among Vietnamese students aged 15 to 25 by investigating environmental factors and pressures. Building
on prior research, the study uses a classification model and analytic techniques to provide a more nuanced understanding
of depression among Vietnamese students.

## 2. Idea

Using machine learning model built on top of classification algorithms, to improve prediction accuracy in various
sectors, particularly in measuring depression levels among Vietnamese students. Advanced algorithms like Support Vector
Machine, Naive Bayes, and K-Nearest Neighbors are used to manage large datasets, identify patterns, and make accurate
predictions.

## 3. Roadmap

- [x] Data collecting
- [x] Data pre-processing
- [x] Build model to classify

Please see the [open issues](https://github.com/dngcphngnh04/DataMining-Project/issues) for a full list of proposed features (
and known issues).

# METHODOLOGY (based on theory class)

## 1. Ask a question

This study focuses on predicting depression levels among Vietnamese students using data mining techniques. It
investigates the accuracy of machine learning algorithms and their practical implications for early detection and
intervention, aiming to improve students' well-being and academic performance.

## 2. Data gathering

The research used a systematic approach to gather data on stress and pressure among Vietnamese students. A **Google Form**
survey was created based on various studies and conversations with university specialists. The survey identified five
main causes of stress: employment, studies, self, family, and love. The Patient Health Questionnaire (PHQ-9) was used to
assess depression risk. The survey was distributed to a diverse group of Vietnamese students, ensuring confidentiality
and anonymity. The data collected was evaluated for quality and completeness, aiming to improve knowledge and early
detection of depression among Vietnamese students.

## 3. Data cleaning

The pre-processing stage of a depression study involved categorizing responses into five groups based on depression-related factors. Python's'shuffle' function ensured no inherent order or bias. Python was used for data processing due to Weka's limitations and Java's difficulties. Missing data was filled using an imputation method, and string-type responses were converted into numerical form using label encoding. The original dataset had long column names, requiring special encoding for quick access. Questions were encoded with alphabetical letters for clarity. This pre-processing made the dataset suitable for further analysis and machine learning methods.

## 4. Model building

Machine learning algorithms are crucial for creating classification models, as they identify trends and make data-driven judgments. Techniques like ibk, J48, Logistic Regression, Naive Bayes, OneR, SVM, AdaBoostM1, RandomForest, and ExtraTree are used to classify instances based on their closest neighbors, making them suitable for pattern recognition tasks, binary classification, and large datasets. Ensemble approaches like AdaBoostM1 and RandomForest increase classification performance by integrating multiple classifiers, while ExtraTree increases resilience by creating decision trees with random splits. Weka supports these models and offers full modeling and training capabilities, making the development process easier. Weka functions like buildClassifier are often used throughout the model development process, making the training process quick for many classifiers.

## 5. Return the result

The final step involves evaluating a machine learning model on a test dataset to assess its effectiveness in identifying and forecasting depression levels. This evaluation uses measures like accuracy, precision, recall, and F1 score, ensuring the model's dependability, accuracy, and generalization capabilities. This step is crucial for enabling proactive early depression detection and management, enhancing student well-being. The model's efficacy is assessed using functions like toSummaryString, toMatrixString, and metrics, providing comprehensive results for a comprehensive study.

# PROJECT STRUCTURE

- `.idea` folder: provides project-specific parameters and configurations for IntelliJ IDEA
- `Data` folder: to hold the datasets organized by family, love, self, study, and job subfolders. Each contains the full training, testing, and validation dataset in both CSV and ARFF format supported by the Weka
- `Evaluation Result + Report` folder: it contains the Evaluation results produced in CSV format and our Report for the Data Mining Project
- `Colab Notebook` folder: the folder contains the Colab Notebook for the Preprocessing Stage of the dataset.
- `lib` folder: contains external libraries and dependencies needed for the project (weka.jar)
- `out` folder: stores compiled output files, such as class files
- `src` folder: the primary source directory for the project’s code
  - `BaseModel` subfolder:
    - `ModelPre_Tuning` subfolder: provides foundational classifier Java classes for this project. That includes IBk, J48, Logistic Regression, Naïve Bayes, OneR, SVM  and ZeroR as default.
    - `TuningModel` subfolder: provides foundational classifier Java classes for this project. That includes IBk, J48, Logistic Regression, OneR and SVM as tuning models.
  - `EnsembleModel` subfolder:
    - `ModelPre_Tuning` subfolder: contains the ensemble model classes that incorporate many classifiers. That includes AdaBoostM1, and RandomForest as default.
    - `TuningModel` subfolder: contains the ensemble model classes that incorporate many classifiers. That includes AdaBoostM1, and RandomForest as tuning models.
  - `Preprocessing` folder:
    - `AttributeSelection.java`: class to import data from the specified path
    - `CSV2Arff.java`: the class for processing and preparing data
    - `DataPreprocess.java`: class to enable feature selection algorithms
- `README.md`: a document to outline and explain the project

# CONTACT

Project Link: **[GitHub HERE](https://github.com/dngcphngnh04/DataMining-Project)**

<!-- ACKNOWLEDGMENTS -->

# ACKNOWLEDGEMENTS

We want to express our sincerest thanks to our lecturer and the people who have helped us to achieve this project's
goals:

- []() Dr. Nguyen Thi Thanh Sang
- []() MSc. Nguyen Quang Phu
- []() The README.md template from **[othneildrew](https://github.com/othneildrew/Best-README-Template)**


<!-- MARKDOWN LINKS & IMAGES -->

[contributors-shield]: https://img.shields.io/github/contributors/dngcphngnh04/DataMining-Project.svg?style=for-the-badge
[contributors-url]: https://github.com/dngcphngnh04/DataMining-Project/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/dngcphngnh04/DataMining-ProjectC.svg?style=for-the-badge
[forks-url]: https://github.com/dngcphngnh04/DataMining-Project/network/members
[issues-shield]: https://img.shields.io/github/issues/dngcphngnh04/DataMining-Project.svg?style=for-the-badge
[issues-url]: https://github.com/dngcphngnh04/DataMining-Project/issues
