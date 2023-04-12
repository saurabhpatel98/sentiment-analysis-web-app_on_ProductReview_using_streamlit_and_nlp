Sentiment Analysis Dashboard on Amazon Product reviews using streamlit and NLP This project provides an interactive dashboard for performing sentiment analysis on product and company reviews. The dashboard is built using Python and Streamlit, and allows users to filter and visualize reviews from a dataset of customer reviews for various products and companies.

The dashboard offers three filters: by company, by product, and by sentiment result. Users can select multiple companies and products to filter the reviews, and can see the filtered data in a table view. The sentiment analysis results are displayed in a pie chart or a bar chart, depending on user preference. The dashboard also provides some key performance indicators (KPIs) such as the total number of reviews, the number and percentage of positive and negative reviews, and the average rating.

The code uses a machine learning model based on Natural Language Processing (NLP) techniques to perform sentiment analysis on the reviews. The model was trained on a dataset of product reviews using Python's Scikit-learn library. The resulting model is then used to classify the reviews as positive or negative.

The dashboard is deployed on Heroku, a cloud platform that allows users to deploy, manage, and scale applications. The source code is available on GitHub, along with the dataset used for training the sentiment analysis model.

Overall, this project provides an easy-to-use and interactive way to analyze customer sentiment towards products and companies. The dashboard can be useful for companies to monitor their online reputation, identify areas for improvement, and make data-driven decisions based on customer feedback.


* Create a virtual environment
  * install virtual environment
 
        pip install virtualenv
        
  * create virtual environment by the name ENV
        
        virtualenv ENV
        
  * activate ENV

        .\ENV\Scripts\activate
        
* Install project dependencies

      pip install -r .\requirements.txt
      
* Run the project

      python app.py
      
* Look for the local host address on Powershell screen, something like: 127.0.0.1:5000 >> Type it on your Web Browser >> Project shall load
* Try out your Amazon Alexa test reviews and look for results
* To close >> Go back to Powershell & type `ctrl+c` >> Deactivate Virtual Environment ENV

      deactivate


### Steps to run on Mac

* Prerequisites: [Python 3.9](https://www.python.org/downloads/)
* Open Terminal >> navigate to working directory >> Clone this Github Repo

      git clone https://github.com/saurabhpatel98/sentiment-analysis-web-app_on_ProductReview_using_streamlit_and_nlp.git  
* Navigate to new working directory (cloned repo folder)
* Create a virtual environment
  * install virtual environment

        pip install virtualenv
        
  * create virtual environment by the name ENV
  
        virtualenv ENV  
  * activate ENV
        
        source ENV/bin/activate
* Install project dependencies

      pip install -r requirements.txt  
* Run the project

      python app.py
      
* Look for the local host address on Terminal screen, something like: 127.0.0.1:5000 >> Type it on your Web Browser >> Project shall load
* Try out your Amazon Alexa test reviews and look for results
* To close >> Go back to Terminal & type `ctrl+c` >> Deactivate Virtual Environment ENV

      deactivate
      
      
![Filtered_data](https://user-images.githubusercontent.com/26132974/231563817-e8edacac-331f-411d-a5df-ae225941a914.png)
![result](https://user-images.githubusercontent.com/26132974/231563901-f963a9d1-7ea4-48ce-a37f-d1dd40a29d94.png)
![Main Page](https://user-images.githubusercontent.com/26132974/231563905-8995483d-c76f-4641-8b94-8a497bd5b938.png)


      

