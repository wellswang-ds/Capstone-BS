# Retail Analytics: fashion and apparel
Capstone Project for BrainStation Data Science Bootcamp
---

## The Problem Area

Generic marketing campaigns may not resonate with individual customers. 

- Can we use information about previous purchases and customer demographics to predict customer lifetime value (CLV) and enable personalized marketing?
- Are we able to predict which customer is going to purchase within the next 10 days?
- Given a new customer, are we able to provide recommendations with the first few items they viewed?

## The User

The benefits of predicting CLV:

- Retailers’ marketing approach could be more personalized with CLV predicted. (ex. targeted promotions, recommendations).
- Retailers can target channels with high-CLV customers thus acquiring customers more efficiently.
- Retailers can better segment customers by value and target different segments with diverse strategies.

## The Big Idea

Predicting customer CLV is a classic yet crucial problem in the retail industry. The preliminary approach design for this project is to:

1. Grouping customers into segments (feature engineering) with RFM analysis (Recency, Frequency, Monetary).
2. Predicting CLV with classic supervised machine learning models (ex. regression).
3. Predicting CLV with deep learning models.
4. Use CLV as a feature, to predict if a customer will purchase in a 10-day window
5. (stretch a little bit here) For new customers, create a prompt where customers answer a few questions or select a few images, then provide purchase recommendations for them. (reinforcement learning?)

**This is going to be modified!**

## The Impact

The predicted CLV can be used in many spectra:

- Identifying high-CLV customers and tailored marketing may increase their spending by 10% to 30%
- Customer lifetime extension by a few months may result in 5% to 15% increase in revenue
- Target customer acquisition based on CLV predictions may reduce acquisition costs by 10%-20%

On the other hand, with the custom recommendation system effective cross-selling may lead to a 5% to 15% boost in revenue as well as enhanced customer loyalty.

## The Data

The data is available from the [H&M Personalized Fashion Recommendation Competition on Kaggle](https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations/data?select=transactions_train.csv)

The dataset consists of:

- 105k image files of the product
- the product (article) data
- the customer data
- the transactions data.

Should the fashion and apparel data not work, a similar idea of CLV could be used on [this Instacart dataset](https://www.kaggle.com/c/instacart-market-basket-analysis/data).

## The Alternative

- **Coffee shop recommendation**: Use computer vision to segment photos taken from a coffee shop (interior, food, coffee), and recommend a similar photo from a different coffee shop (recommendation based on “vibes” and products). (Might need a lot of data collection)
- **Restaurant Sales and Weather**: Restaurant sales seem to be affected a lot by the weather. Are we able to estimate the traffic of certain venues considering the weather to support decision-making for merchants? (might need some web-scraping for the venue traffic, and data collection of the restaurant sales)