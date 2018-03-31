Project name : Titanic: Machine Learning from Disaster.
Leader Board Score on Kaggle: 0.79904 

Project description: 
The Titanic Dataset from kaggle has been around the internet since 2012.  

Start asking this questions and answers will  help to increase accuracy.
Questions to address
1) who were the passengers?
2) Which class passengers belong?
3) what deck were the passengers in?
4) where did the passengers come from?
5) Number of people alone and with family

Well we dont need the exact count, we just need overview.

Points I noticed in this data_set are as follows.
1.Number of Male passengers is more than female passengers.
2.Number of passengers in class 3 are more than class 1 , class 2.
3.Cabin B,C had more number of passengers as compared to other cabins.
4.High number of passengers are Embarked from port S = Southampton.
5.Alone people are more than people with family.

I used seaborn visualization library to visualize the data.

When it comes to survival I observed,
1)Children and Female are saved first that's why survival rate of Children and Female is more than survival rate of Male's.
2)Survival rate in 3rd class is much lower.
3)The survival rate of male's in the ship reduces with age whereas for women it is higher.
4)The survival rate of people with family is considerably higher when compared to the people without family.

Considering this observation did some Feature_Engineering which can be seen in the code.

After using various models it is notised that RandomForest is giving good accuracy.

Evaluated with F1_score, ROC_AUC score.
