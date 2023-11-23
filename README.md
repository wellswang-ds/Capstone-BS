# Demand Forecasting and Trending Item Prediction for H&M Retail Sales
Capstone Project for BrainStation Data Science Bootcamp
---

## The Problem Area

Fast-fashion companies struggle to control inventory. 

- By predicting trending items within a few days of launch, companies can respond to high demands in inventory. 
- This project aims to utilize H&M sales data to predict trending items after a certain period since launch. By analyzing the sales history of H&M products, we can identify patterns and trends that can be used to make informed predictions about which products are likely to become popular in the future.
- The fashion industry is constantly changing, and it can be difficult for companies to keep up with the latest trends. By predicting which products are likely to become popular, H&M can stay ahead of the competition and ensure that they are offering their customers the latest and most desirable products.

## The User

The benefits of predicting trending items:

- Retailers’ marketing approach could be more personalized with trending items predicted. (ex. targeted promotions, recommendations).
- We can understand the features of the trending items, and provide this information to the design, strategy team to improve customer loyalty.
- Inventory management can be greatly improved as we can get a hint of what's going to be selling a lot.

## The Big Idea

To actually help with inventory management, we need to define a response time from ordering to re-stock. The assumption of the project has set the response time to 1 month. (i.e. we can predict an item would be trending in a month, then we can order more a month before)

We have 2 approach to this problem:

1. Binary Classification: Given input of n days of sale of a product (with the features included), can we predict if it is going to reach a certain sales threshold (trending)?

2. Demand Forecasting + Rule: We can create a lag 30 day time series prediction model. After the model is created, we can add rules to specify anomolies (trending sales).


## The Impact

- Increased Sales and Revenue:
By predicting trending items after a certain period since launch, H&M can ensure that they have enough stock to meet demand. This will lead to increased sales and revenue for the company.

- Improved Customer Satisfaction
By having the trending items in stock, H&M can improve customer satisfaction by ensuring that customers can find what they are looking for. This will lead to increased customer loyalty and repeat business.


## The Data

The data is available from the [H&M Personalized Fashion Recommendation Competition on Kaggle](https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations/data?select=transactions_train.csv)

The dataset consists of:

- 105k image files of the product
- the product (article) data
- the customer data
- the transactions data.

We've cleaned and merged the datasets. Here's the merged columns after cleaning:
 #   Column                        Dtype  
---  ------                        -----  
 0   customer_id                   object 
 1   article_id                    int64  
 2   unit_price                    float64
 3   units                         int64  
 4   sales_channel                 object 
 5   prod_name                     object 
 6   product_type_name             object 
 7   product_group_name            object 
 8   graphical_appearance_name     object 
 9   colour_group_name             object 
 10  perceived_colour_value_name   object 
 11  perceived_colour_master_name  object 
 12  department_name               object 
 13  index_name                    object 
 14  index_group_name              object 
 15  section_name                  object 
 16  garment_group_name            object 
 17  detail_desc                   object 
 18  FN                            bool   
 19  Active                        bool   
 20  club_member_status            object 
 21  fashion_news_frequency        object 
 22  age                           float64
 23  postal_code                   object 
dtypes: bool(2), float64(2), int64(2), object(18)
memory usage: 5.0+ GB

