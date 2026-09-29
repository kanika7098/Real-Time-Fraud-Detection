 Real-Time Online Payment Fraud Detection



This is a machine learning project I made to detect fraudulent online payment transactions.



The main idea is to enter transaction details, predict the probability of fraud, and show a risk score and risk level.



I used the PaySim synthetic financial transaction dataset for this project. I trained a Random Forest model and connected it with FastAPI and Streamlit to make a working application.



\## What this project does



\* Predicts fraud probability

\* Calculates a risk score from 0 to 100

\* Shows Low, Medium, or High risk

\* Gives some reasons for the risk

\* Provides a FastAPI prediction API

\* Provides a Streamlit dashboard

\* Shows model performance

\* Shows SHAP feature importance

\* Keeps recent transaction results during a dashboard session



\## Technologies Used



\* Python

\* Pandas

\* NumPy

\* Scikit-learn

\* Random Forest

\* SHAP

\* FastAPI

\* Streamlit

\* Matplotlib

\* Joblib

\* Jupyter Notebook



\## Dataset



I used the PaySim synthetic financial transaction dataset.



The raw dataset is not uploaded to GitHub because the CSV file is very large.



Place the dataset inside:



```text

data/raw/

```



\## Machine Learning



I explored the dataset first and checked things such as:



\* missing values

\* duplicate values

\* transaction types

\* transaction amounts

\* fraud distribution



I then created features for the model.



For the final deployed model, I used features that can be available when making the transaction decision.



The final model is a Random Forest classifier.



\## Risk Score



The model gives a fraud probability.



I convert this probability into a score from 0 to 100.



For example:



```text

Fraud Probability = 14.67%

Risk Score = 14.67

```



The dashboard uses these risk rules:



```text

0 to <5     = Low

5 to <50    = Medium

50 to 100   = High

```



\## SHAP Explainability



I used SHAP to understand which features have more influence on the model predictions.



The dashboard contains a separate Explainability section for this.



\## FastAPI



I created a FastAPI backend for the trained model.



The main endpoint is:



```text

POST /predict

```



The API returns:



\* fraud probability

\* risk score

\* risk level

\* risk reasons



FastAPI also provides Swagger documentation for testing the API.



\## Streamlit Dashboard



I created a Streamlit dashboard where I can enter transaction details manually.



The dashboard shows:



\* fraud probability

\* risk score

\* risk level

\* risk indicators

\* recent transaction results

\* model performance

\* SHAP feature importance



I also added a known fraud example for testing.



\## Example Results



One transaction I tested gave:



```text

Fraud Probability: 14.67%

Risk Score: 14.67 / 100

Risk Level: Medium

```



Another transaction gave:



```text

Fraud Probability: 100.00%

Risk Score: 100.00 / 100

Risk Level: High

```



I also tested manually entered transaction values through the dashboard.



\## Project Structure



```text

Real-Time-Fraud-Detection/

│

├── app/

│   └── app.py

│

├── data/

│   └── raw/

│

├── models/

│

├── notebooks/

│   └── 01\_data\_exploration.ipynb

│

├── screenshots/

│

├── dashboard.py

├── requirements.txt

├── .gitignore

└── README.md

```



\## How to Run



Install the required packages:



```bash

python -m pip install -r requirements.txt

```



Start FastAPI:



```bash

python -m uvicorn app.app:app --reload

```



Open:



```text

http://127.0.0.1:8000/docs

```



Then open another Command Prompt and run:



```bash

python -m streamlit run dashboard.py

```



Open:



```text

http://localhost:8501

```



\## Limitations



This project uses synthetic data, so the results should not be treated as results from a real banking system.



The risk thresholds are configurable rules used by the application.



The model would need to be properly retrained and validated on representative real-world data before being used for actual financial decisions.



\## Future Improvements



\* Better threshold tuning

\* Model comparison

\* Cloud deployment

\* Database integration

\* Automatic alerts

\* Transaction monitoring over time

\* Model drift monitoring



\## Author



Kanika Gaikwad



