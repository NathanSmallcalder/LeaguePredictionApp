import mysql.connector
import sys
import seaborn as sns
sys.path.append('../..')
from config import *
from RiotApiCalls import *

# Data Processing
import pandas as pd
import numpy as np

# Modelling
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,  classification_report
from sklearn.model_selection import  train_test_split
from sklearn.metrics import mean_squared_error, log_loss
from sklearn.metrics import RocCurveDisplay
import matplotlib.pyplot as plt

def create_connection():
        # Database configuration
        config = {
            'user': 'root',
            'password': password,
            'host': 'mysql-db',
            'port': 3306,
            'database': 'LeagueStats',
        }

        # Connect to the database
        connection = mysql.connector.connect(**config)
        return connection

global rf
 
def randomForestRun():
    query = ("SELECT `SummonerMatchTbl`.ChampionFk, `MatchStatsTbl`.`MinionsKilled`,`MatchStatsTbl`.`kills`,`MatchStatsTbl`.`deaths`,`MatchStatsTbl`.`assists`,`MatchStatsTbl`.Lane,  `MatchStatsTbl`.CurrentMasteryPoints, `MatchStatsTbl`.`DmgDealt`,`MatchStatsTbl`.`DmgTaken`,`MatchStatsTbl`.`TurretDmgDealt`,`MatchStatsTbl`.`TotalGold`,`MatchStatsTbl`.EnemyChampionFk,  `MatchTbl`.`GameDuration`,`MatchStatsTbl`.`DragonKills`,`MatchStatsTbl`.`BaronKills` ,`MatchStatsTbl`.`Win` FROM `SummonerMatchTbl` JOIN `MatchStatsTbl`ON `MatchStatsTbl`.SummonerMatchFk = `SummonerMatchTbl`.SummonerMatchId JOIN `MatchTbl` ON `MatchTbl`.`MatchId` = `SummonerMatchTbl`.`MatchFk` WHERE `MatchTbl`.`QueueType` = 'CLASSIC';")
    connection = create_connection()
    cursor =  connection.cursor()
    cursor.execute(query)
    data = cursor.fetchall()

    columns = ['ChampionFk', 'MinionsKilled','kills','deaths','assists','lane','CurrentMasteryPoints',
    'DmgDealt','DmgTaken','TurretDmgDealt','TotalGold'
    ,'EnemyChampionFk', 'GameDuration','DragonKills','BaronKills','Win']

    #data = pd.read_csv("data.csv")
    df_games = pd.DataFrame(data,columns = columns)
    df_games.sample(frac=1)
    print(df_games.head())


    df_games['lane'] = df_games['lane'].map({'TOP':0,'JUNGLE':1,'MIDDLE':2,'BOTTOM':3,'SUPPORT':4,'NONE':5})
    df_games['Win'] = df_games['Win']
    X = df_games.drop('Win', axis=1)
    y = df_games['Win']
    print(df_games.head())
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

    global rf
    rf = RandomForestClassifier()
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    clf_probs = rf.predict_proba(X_test)
    print("Accuracy:", accuracy)
    print(classification_report(y_test, y_pred))
    score = log_loss(y_test, clf_probs)
    print("Log Loss", score)
    mse = mean_squared_error(y_test, y_pred)
    print("MSE: ", mse)

    corr_matrix =  X.corr()
    print(corr_matrix)

    plt.figure(figsize=(16,14))
    plt.title('Correlation Heatmap of riot Dataset')
    a = sns.heatmap(corr_matrix, square=True, annot=True, fmt='.2f', linecolor='black')
    a.set_xticklabels(a.get_xticklabels(), rotation=30)
    a.set_yticklabels(a.get_yticklabels(), rotation=30)           
    plt.savefig("figure.png")
    #v_threashhold = VarianceThreshold(threshold=0)
    #v_threashhold.fit(X)
    #print(v_threashhold.get_support())

    return rf

def randomForestPredict(rf, ChampionFk,MinionsKilled,kills,deaths,assists,lane,CurrentMasteryPoints,DmgDealt,DmgTaken,TurretKills,TotalGold,EnemyChampionFk,GameDuration,DragonKills,BaronKills):
    #y_pred = rf.predict(X_test)
    #accuracy = accuracy_score(y_test, y_pred)
    #print("Accuracy:", accuracy)
    #'ChampionFk', 'MinionsKilled','kills','deaths','assists','lane','DmgDealt','DmgTaken','TurretDmgDealt','TotalGold' 'EnemyChampionFk', 'GameDuration','DragonKills','BaronKills',
    row = [[ChampionFk,MinionsKilled ,kills,deaths,assists,lane,CurrentMasteryPoints,DmgDealt,DmgTaken,TurretKills,TotalGold,EnemyChampionFk,GameDuration,DragonKills,BaronKills]]
    prob = rf.predict_proba(row)
    yhat = rf.predict(row)
    print('Prediction: %d' % yhat[0],)
    prediction = {
         "pred":yhat[0],
         "probability":[prob[0][0],prob[0][1]]
    }
    print(prediction['probability'])
    return prediction

def getRandomForest():
    randomForestRun()
    global rf
    return rf

def getPlotScore(rf, X_test, y_test):
    svc_disp = RocCurveDisplay.from_estimator(rf, X_test, y_test)
    return svc_disp