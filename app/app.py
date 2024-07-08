from flask import Flask, render_template, request, redirect, url_for, session, jsonify
import requests
import secrets
from RiotApiCalls import *
import pandas as pd
import matplotlib.pyplot as plt 
import numpy as np
import warnings
from championsRequest import *
import sys
import json
import databaseQuries as databaseQuries
from mlAlgorithms import *
from mlAlgorithms.SoloPredictor import randomForestSolo
from mlAlgorithms.TeamPredictor import randomForest
from databaseQuries import *
import config

warnings.filterwarnings('ignore')

app = Flask(__name__)

### Routing for GET summonerData
### gets Summoner Data from last 5 games
### Converts to a JSON response, returned to the matchPredict endpoint, which sends to the PredictSolo endPoint for prediction
@app.route('/summData', methods = ['GET'])
def summData():
    summonerName = request.args.get('summoner')
    Region = request.args.get('region')
    champ = request.args.get('champ')
    RegionStart = "europe"
    enemyChamp = request.args.get('enemyChamp')
    lane = request.args.get('lane')
    tagline = request.args.get('tagline')

    Summoner = getPuuid(RegionStart,summonerName,tagline)

    puuid = Summoner['puuid']
    print(puuid)
    SummonerInfo = getSummonerDetails(Region,puuid)
    print(SummonerInfo)
    SummId = SummonerInfo['id']

    print(champ)
    ### Gets Mastery Stats
    mastery = getMasteryStats(Region, puuid)
    RankedDetails = getRankedStats(Region,SummId)
    data = getMatchData5Matches(Region, puuid, SummonerInfo,RankedDetails,mastery)

    mastery = getSingleMasteryScore(champ, mastery)
    

    ### Find Avg Stats for previous games
    avg = AvgStats(data)
    ### Set Other Values for the Array
    avg['ChampId'] = champ
    avg['masteryPoints'] = mastery
    avg['enemyChamp'] = enemyChamp
    avg['lane'] = lane
        
    return jsonify(avg), 200

### Routing for Match Predictor (SOLO)
### Expects a JSON request of 
###    json_data = { 
###     "MinionsKilled": 258,
###     "kills": 25,
###     "assists": 56,
###     "deaths": 1,
###     "TotalGold": 32355,
###     "DmgDealt": 422425,
###     "DmgTaken": 24567,
###     "DragonKills": 4,
###     "BaronKills": 3,
###     "GameDuration": 200,
###      "TurretDmgDealt": 4,
###      "ChampionFk": 1,
###      "masteryPoints": 42257,
###      "EnemyChampionFk":2 ,
###      "lane": 1
###      }
### Calls randomForest Predict and runs the data through the algorithm
### Returns Prediction JSON - Value 1 or 0
@app.route('/predictSolo',methods=['POST','GET'])
def predict():
    content_type = request.headers.get('Content-Type')
    if (content_type == 'application/json'):
     
        data = json.loads(request.data)    

        rf = randomForestSolo.randomForestRun()
        rf = randomForestSolo.randomForestPredict(rf,data['ChampionFk'],data['MinionsKilled'],data['kills'],data['deaths'],data['assists'],data['lane'], data['masteryPoints'],
                                            data['DmgDealt'],data['DmgTaken'],data['TurretDmgDealt'],data['TotalGold'],data['EnemyChampionFk'],
                                            data['GameDuration'],data['DragonKills'],data['BaronKills'])
        if (int(rf["pred"]) == 0):
            rf['pred'] = "Loss"
        else:
            rf['pred'] = "Win"

        print(rf['probability'])

        percent = [int(rf['probability'][0] * 100), int(rf['probability'][1] * 100)]
        Prediction = {
            "pred": str(rf['pred']),
            "percentage": percent
        }

        print(Prediction)
        return jsonify(Prediction), 200
    else:
        return "Invalid Request Data, Make Sure all summoners exist",500
    

@app.route('/matchPredict', methods = ['GET','POST'])
def matchPredict():
    champ = getAllChampions()
    print(champ)
    champ.pop(0)
    RoleImages = getRoles()
    return render_template('matchPrediction.html', Champions = champ, RoleSelect = RoleImages)

### Routing for the /teamPredict Endpoint
### Returns UI for teamPrediction
### Users can input variables for teams
@app.route('/teamPredict',methods = ['GET'])
def teamPredictor():
    RoleImages = getRoles()
    champ = getAllChampions()
    print(champ)
    return render_template('teamMatchPrediction.html', RoleImages = RoleImages, Champions = champ)

### Routing Endpoint for Team Prediction
### Excpects the following JSON format
###   dataset = {
###        "B1Summ": "Mealsz",
###        "B2Summ": "Ehhhh",
###        "B3Summ": "Itwoznotmee",
###        "B4Summ": "Lil Nachty",
###        "B5Summ": "Forza Napøli ",
###        "R1Summ": "Primabel Ayuso",
###        "R2Summ": "NateNatilla",
###        "R3Summ": "sweet af",
###        "R4Summ": "Fedy9 ",
###        "R5Summ": "ChampagneCharlie ",
###        "B1": 44,
###        "B2": 876,
###        "B3": 136,
###        "B4": 221,
###        "B5": 74,
###        "R1": 122,
###        "R2": 20,
###        "R3": 99,
###        "R4": 202,
###        "R5": 412,
###        'Region':"EUW1"
###    }
### Converts Summoner names into desiered team (blue or red) 
### Runs both teams through calculateAvgTeamStats
### Makes the Dataset to be ran through the machine learning algorithm
### returns prediction json
###    {
###     'BlueTeam': x
###      'RedTeam': y 
###   }
@app.route('/teamData', methods = ['GET','POST'])
def teamData():
    content_type = request.headers.get('Content-Type')
    if (content_type == 'application/json'):
        data = json.loads(request.data)
        region = "EUW1"

        BlueTeam = [[str(data['BlueSumm1']),str(data['Bluetag1']),data['blueSumm1Champ']],
                    [str(data['BlueSumm2']),str(data['Bluetag2']),data['blueSumm2Champ']],
                    [str(data['BlueSumm3']),str(data['Bluetag3']),data['blueSumm3Champ']],
                    [str(data['BlueSumm4']),str(data['Bluetag4']),data['blueSumm4Champ']],
                    [str(data['BlueSumm5']),str(data['Bluetag5']),data['blueSumm5Champ']]],
    
        RedTeam = [[str(data['RedSumm1']),str(data['Redtag1']),data['redSumm1Champ']],
                    [str(data['RedSumm1']),str(data['Redtag2']),data['redSumm2Champ']],
                    [str(data['RedSumm3']),str(data['Redtag3']),data['redSumm3Champ']],
                    [str(data['RedSumm4']),str(data['Redtag4']),data['redSumm4Champ']],
                    [str(data['RedSumm5']),str(data['Redtag5']),data['redSumm5Champ']]],
        

        blueTeam = calculateAvgTeamStats(BlueTeam,region)
        redTeam = calculateAvgTeamStats(RedTeam, region)
    
        dataSet = makeDataSet(blueTeam,redTeam,data)
      
        rf = randomForest.randomForestMultiRun()
        prediction = randomForest.randomForestPredictMulti(rf,dataSet)
        pred = {
            "Blue": None,
            "Red": None,
            "Probability": [prediction['probability'][0] *100, prediction['probability'][1] * 100]
        }

        if int(prediction['RedTeam']) == 0:
            pred['Blue'] = 'Win'
            pred['Red'] = 'Loss'
        else:
            pred['Blue'] = 'Loss'
            pred['Red'] = 'Win'
    
        print(pred)
        return pred ,200
    else:
        return "Invalid Request Data, Make Sure all summoners exist", 500
    
### Routing for the Main page
@app.route('/')
def index():
    return render_template('index.html')

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8001)