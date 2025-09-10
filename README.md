# Medical_Insurance_cost_prediction
Of course\! Here is a brief and informative description of your project, perfect for your GitHub README file.

-----

# Medical Insurance Prediction System

This project is a web-based application designed to predict medical insurance costs and assess health risks for users. By leveraging machine learning, it provides personalized estimates based on individual health and demographic data. The system features a secure user authentication portal and an intuitive interface for a seamless user experience.

##  Core Features

  * **Insurance Cost Prediction**: Utilizes a **Linear Regression** model to accurately estimate a user's medical insurance premiums based on inputs like age, BMI, number of children, and smoking status.
  * **Health Risk Assessment**: Employs a **Random Forest Classifier** to predict a user's risk of diabetes. This prediction is then combined with other health factors (like high BMI or smoking) to classify their overall health risk into **Low**, **Medium**, or **High** categories.
  * **Interactive Web Application**: A user-friendly dashboard built with **Streamlit** that allows users to easily input their information and view their personalized predictions instantly.
  * **Secure User Authentication**: A complete login and registration system is implemented using **SQLite** to store user credentials and **bcrypt** for secure password hashing.

## 🛠️ Technology Stack

  * **Backend & Machine Learning**: Python, Scikit-learn, Pandas
  * **Frontend**: Streamlit
  * **Database**: SQLite
  * **Models Used**:
      * Linear Regression (for Insurance Cost)
      * Random Forest Classifier (for Diabetes/Health Risk)

## 💡 How to Run the Project

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/shriya-shree/Medical_Insurance_cost_prediction.git
    cd your-repository-name
    ```

2.  **Install the dependencies:**

    ```bash
    pip install -r requirements.txt
    ```


3.  **Run the Streamlit application:**

    ```bash
    streamlit run app1.py
    ```

-----

### Suggestions for Enhancement

  * **Model Performance Dashboard**: Add a separate page in the Streamlit app to visualize the performance metrics (e.g., R-squared for regression, confusion matrix for classification) of your trained models.
  * **Data Visualization**: Display charts and graphs of the user's input data compared to the training dataset's distribution to give them more context about their health profile.
  * **Expand Risk Factors**: Incorporate more health conditions beyond diabetes into the risk assessment model to provide a more comprehensive health overview.
  * **Deployment**: Deploy the application to a cloud service like Heroku, AWS, or Streamlit Cloud so that it's publicly accessible.
