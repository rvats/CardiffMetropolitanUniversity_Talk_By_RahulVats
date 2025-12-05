# Predicting California Wildfire Likelihood Utilizing Historical Weather and Fire Data

### In Collaboration With

- [Jeff Warchall](https://jeffwarchall.com/)
- [Adam Swan](https://acswan9690.github.io/)

---

### Problem Statement

To help the California Department of Forestry and Fire Protection allocate resources, can we predict the likelihood of fires utilizing historical weather and wildfire data?

---

### Executive Summary

Machine learning algorithms are some of the most powerful and interesting tools in tech right now and their applications are nearly limitless.  From predicting stock prices to determining how likely someone is to have cancer, we can use them to help us anticipate the future.  In this project, our team sought to predict whether there would be a wildfire in locations of California using historical fire and weather data between 2008 and 2020.  We know weather conditions are highly colinear in that every variable changes as a direct result of another.  Fire likelihood is also a direct result of weather conditions in that heat usually leads to dryness and dryness leads to more ignition sources resulting in an elevated fire risk.  Because of these correlations we wanted to see if we could predict whether I fire is likely to occur somewhere using this historical data.


**1. Sample details:**
- 10,988 total entries from every month between 2008 - 2020
- Multi-indexing for instances of multiple fires at different locations occurring in the same month and year
- 4,279 occurrences of fire within our combined dataset
- 6,709 entries without record of a fire starting

**2. Sources:**
- https://gis.data.ca.gov/
- https://data.ca.gov/
- https://mygeodata.cloud/converter/
- https://www.worldweatheronline.com/
- https://towardsdatascience.com/


**3. Data Details:**


|Feature|Type|Description|
|---|---|---|
|**date**|*object*|The month and year of when the fire took place.|
|**county**|*object*|The county the fire started in.|
|**maxtempF**|*float*|The average maximum temperature of that month in °F.|
|**mintempF**|*float*|The average min temperature of that month in °F.|
|**avgtempF**|*float*|The monthly average, in °F, of the daily average temperature for that month .|
|**totalSnow**|*float*|The total snow for that month.|
|**humid**|*float*|The average humidity for that month.|
|**wind**|*float*|The average wind speed for that month.|
|**precip**|*float*|The total precipitation for that month.|
|**q_avgtempF**|*float*|The quarterly average temperature in °F.|
|**q_avghumid**|*float*|The quarterly average humidity.|
|**q_sumprecip**|*float*|The quarterly total precipitation.|
|**sunHour**|*float*|The average hours of sun for that month.|
|**FIRE_NAME**|*object*|The name of the fire.|
|**CAUSE**|*float*|The cause of the fire.|
|**lat**|*float*|The latitude coordinate of the center of the county in which the fire was located.|
|**long**|*float*|The longitude coordinate of the center of the county in which the fire was located.|
|**GIS_ACRES**|*float*|The total number of acres burned.|



**4. Target:**

Four classification models were used: Logistic Regression, KNN Classifier, Random Forest Classifier, and Voting Classifier utilizing a KNN Classifier, Random Forest Classifier, and Ada Boost Classifier.


**5. Model Performance:**

A total of four models were fit using many different parameters.  Listed below are the two best metrics for each model, further metrics are used for evaluation in the modeling notebook.

| Model Type  | Metric  | Score  |
|---|---|---|
| **Logistic Regression**   | *Accuracy*  | 76%  |
| **Logistic Regression**  | *Precision*  | 67%  |
| **Random Forest Classifier**  | *Accuracy*  | 88% |
| **Random Forest Classifier** | *Precision*  | 84% |
| **Voting Classifier** | *Accuracy* | 87% |
| **Voting Classifier** | *Recall*   | 86% |


---

### Conclusion & Recommendations

Our models were able to predict whether a fire would occur with a surprising level of accuracy.  Out best overall model achieved 88% accuracy while our second best was close behind with a great recall score of 86%.  Due to the danger of wild fires growing out of control and the threat of climate change looming, we recommend the use of our Random Forest model as it positively predicts more fires.  However, it also predicts more false positives meaning there's a higher chance for predicting fires that don't actually happen.  This is where you, the California Department of Forestry and Fire Protection, must prioritize what is most important to you and whether or not you have enough resources and man power to use the above model.

If resources are particularly thin the Voting Classifier may be best as it resulted in fewer false positives which would minimize wasted time and resources at the expense of possibly missing out on fires somewhere else.

---