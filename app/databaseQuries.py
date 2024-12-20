import sys
from championsRequest import *
from config import *
from RiotApiCalls import *
from championsRequest import *
import mysql.connector
import json


#Creates a connection to database
def create_connection():
    config = {
    'user': 'root',
    'password': password,
    'host': 'mysql-db',
    'port': 3306,
    'database': 'LeagueStats',
    }
    connection = mysql.connector.connect(**config)
    return connection

    
def winLossRatio():
    connection = create_connection()
    cursor = connection.cursor()

    # Define the one-line query to get champions with the highest win to loss ratio, limited to 15
    query = "SELECT c.ChampionName, SUM(m.Win = 1) AS Wins, COUNT(m.MatchStatsId) AS TotalGames FROM SummonerMatchTbl s JOIN MatchStatsTbl m ON s.SummonerMatchId = m.SummonerMatchFk JOIN ChampionTbl c ON s.ChampionFk = c.ChampionId GROUP BY s.ChampionFk;"

    # Execute the query
    cursor.execute(query)
    Champions = []
    results = cursor.fetchall()
    for (ChampionName, Wins, TotalGames) in results:
        # Calculate win to loss ratio; handle division by zero
        WinRatio = Wins / (TotalGames or 1)  # Default to 1 if Losses is 0 to avoid division by zero
        Champions.append({
            "ChampionName": ChampionName,
            "Wins": Wins,
            "Losses": TotalGames,
            "WinLossRatio": round(WinRatio, 2) * 100 # Round to 2 decimal places
        })

    Champions.sort(key=lambda x: x["Wins"], reverse=True)

    # Limit to top 15 champions
    Champions = Champions[:5]
    

    return Champions


def totalWinsChampsAll():
    connection = create_connection()
    cursor = connection.cursor()
    # Define the query on a single line with a LIMIT of 15
    query = "SELECT c.ChampionName AS ChampionName, COUNT(CASE WHEN m.Win = 1 THEN 1 END) AS WinCount FROM SummonerMatchTbl s JOIN MatchStatsTbl m ON s.SummonerMatchId = m.SummonerMatchFk JOIN ChampionTbl c ON s.ChampionFk = c.ChampionId GROUP BY s.ChampionFk LIMIT 15;"

    # Execute the query
    cursor.execute(query)
    results = cursor.fetchall()
    data = []

    # Populate the list with the results
    for (Champion, WinCount) in results:
        data.append({"ChampId": Champion, "Win Count": WinCount})

    return data

#Gets Count of Total Games from a given championId
#e.g {'COUNT(`MatchStatsTbl`.Win)': 500}
def totalGames(champId): 
    connection = create_connection()
    cursor =  connection.cursor()
    TotalGames = cursor.execute("SELECT COUNT(`MatchStatsTbl`.Win) FROM `MatchStatsTbl` JOIN `SummonerMatchTbl` on `MatchStatsTbl`.MatchStatsId = `SummonerMatchTbl`.`SummonerMatchId` JOIN `MatchTbl` on `SummonerMatchTbl`.`MatchFk` = `MatchTbl`.`MatchId` WHERE `SummonerMatchTbl`.`ChampionFk` = %s", (champId), )
    TotalGames = cursor.fetchall()
    TotalGames = TotalGames[0]['COUNT(`MatchStatsTbl`.Win)']
    return TotalGames

#Gets Count of Total Games from a given championId and SummonerId
#e.g {'COUNT(`MatchStatsTbl`.Win)': 500}
def totalGamesSummoner(champId,SummonerFk): 
    connection = create_connection()
    cursor =  connection.cursor()
    TotalGames = cursor.execute("SELECT COUNT(`MatchStatsTbl`.Win) FROM `MatchStatsTbl` JOIN `SummonerMatchTbl` on `MatchStatsTbl`.MatchStatsId = `SummonerMatchTbl`.`SummonerMatchId` JOIN `MatchTbl` on `SummonerMatchTbl`.`MatchFk` = `MatchTbl`.`MatchId` WHERE `SummonerMatchTbl`.`ChampionFk` = %s and SummonerMatchTbl.SummonerFk = %s", (champId, int(SummonerFk)))
    TotalGames = cursor.fetchall()
    TotalGames = TotalGames[0]['COUNT(`MatchStatsTbl`.Win)']
    return TotalGames

#Gets Count of Wins  from a given championId by rank
#e.g {'COUNT(`MatchStatsTbl`.Win)': 500}
def champWins(champId): 
    connection = create_connection()
    cursor =  connection.cursor()
    ChampWins = cursor.execute("SELECT COUNT(`MatchStatsTbl`.Win) FROM `MatchStatsTbl` JOIN `SummonerMatchTbl` on `MatchStatsTbl`.MatchStatsId = `SummonerMatchTbl`.`SummonerMatchId` JOIN `MatchTbl` on `SummonerMatchTbl`.`MatchFk` = `MatchTbl`.`MatchId` WHERE `MatchStatsTbl`.Win = 1 and `SummonerMatchTbl`.`ChampionFk` = %s", (champId, ))
    ChampWins = cursor.fetchall()
    ChampWins = ChampWins[0]['COUNT(`MatchStatsTbl`.Win)']
    return ChampWins

#Gets Count of Wins from a given championId and SummonerId
#e.g {'COUNT(`MatchStatsTbl`.Win)': 500}
def champWinsSummoner(champId,SummonerFk): 
    connection = create_connection()
    cursor =  connection.cursor()
    ChampWins = cursor.execute("SELECT COUNT(`MatchStatsTbl`.Win) FROM `MatchStatsTbl` JOIN `SummonerMatchTbl` on `MatchStatsTbl`.MatchStatsId = `SummonerMatchTbl`.`SummonerMatchId` JOIN `MatchTbl` on `SummonerMatchTbl`.`MatchFk` = `MatchTbl`.`MatchId` WHERE `MatchStatsTbl`.Win = 1 and `SummonerMatchTbl`.`ChampionFk` = %s and SummonerMatchTbl.SummonerFk = %s", (champId, int(SummonerFk)))
    ChampWins = cursor.fetchall()
    ChampWins = ChampWins[0]['COUNT(`MatchStatsTbl`.Win)']
    return ChampWins

#Gets Sum of Kills from a given championId by rank
#e.g {'SUM(`MatchStatsTbl`.kills)': 500}
def champKills(champId): 
    connection = create_connection()
    cursor =  connection.cursor()
    ChampKills = cursor.execute("SELECT SUM(`MatchStatsTbl`.kills) FROM `MatchStatsTbl` JOIN `SummonerMatchTbl` on `MatchStatsTbl`.MatchStatsId = `SummonerMatchTbl`.`SummonerMatchId` JOIN `MatchTbl` on `SummonerMatchTbl`.`MatchFk` = `MatchTbl`.`MatchId` WHERE `SummonerMatchTbl`.`ChampionFk` = %s", (champId))
    ChampKills = cursor.fetchone()
    ChampKills = ChampKills['SUM(`MatchStatsTbl`.kills)']
    return ChampKills


#Gets average Avg Minions Per Rank
#e.g {'Rank':'unranked' , 'AVG(`MatchStatsTbl`.`MinionsKilled`)': 500}
def avgMinionsAll(): 
    connection = create_connection()
    cursor =  connection.cursor()
    MinionsAvg = cursor.execute("SELECT `RankTbl`.`RankName` , AVG(`MatchStatsTbl`.`MinionsKilled`) FROM `MatchStatsTbl` JOIN `SummonerMatchTbl` on `MatchStatsTbl`.MatchStatsId = `SummonerMatchTbl`.`SummonerMatchId` JOIN `MatchTbl` on `SummonerMatchTbl`.`MatchFk` = `MatchTbl`.`MatchId` JOIN `RankTbl` on `RankTbl`.`RankId` = `MatchTbl`.`RankFk` WHERE `MatchTbl`.QueueType = 'CLASSIC' GROUP by `MatchTbl`.`RankFk`")
    MinionsAvg = cursor.fetchall()
    return MinionsAvg

###Gets the AvgMinions of a give Summoner by ID
def avgMinionsSummonerAll(SummonerFk): 
    connection = create_connection()
    cursor =  connection.cursor()
    MinionsAvg = cursor.execute("SELECT `RankTbl`.`RankName` , AVG(`MatchStatsTbl`.`MinionsKilled`) FROM `MatchStatsTbl` JOIN `SummonerMatchTbl` on `MatchStatsTbl`.MatchStatsId = `SummonerMatchTbl`.`SummonerMatchId` JOIN `MatchTbl` on `SummonerMatchTbl`.`MatchFk` = `MatchTbl`.`MatchId` JOIN `RankTbl` on `RankTbl`.`RankId` = `MatchTbl`.`RankFk` WHERE `MatchTbl`.QueueType = 'CLASSIC' and SummonerMatchTbl.SummonerFk = %s GROUP by `MatchTbl`.`RankFk`", (int(SummonerFk)))
    MinionsAvg = cursor.fetchall()
    return MinionsAvg

#Gets average Avg Minions of champion from a given championId by rank
#e.g {'Rank':'unranked' , 'AVG(`MatchStatsTbl`.`MinionsKilled`)': 500}
def avgMinions(champId): 
    connection = create_connection()
    cursor =  connection.cursor()
    MinionsAvg = cursor.execute("SELECT `RankTbl`.`RankName` , AVG(`MatchStatsTbl`.`MinionsKilled`) FROM `MatchStatsTbl` JOIN `SummonerMatchTbl` on `MatchStatsTbl`.MatchStatsId = `SummonerMatchTbl`.`SummonerMatchId` JOIN `MatchTbl` on `SummonerMatchTbl`.`MatchFk` = `MatchTbl`.`MatchId` JOIN `RankTbl` on `RankTbl`.`RankId` = `MatchTbl`.`RankFk` WHERE `MatchTbl`.QueueType = 'CLASSIC' AND `SummonerMatchTbl`.`ChampionFk` = %s GROUP by `MatchTbl`.`RankFk`",(champId))
    MinionsAvg = cursor.fetchall()
    return MinionsAvg

#Gets average Avg Minions of champion from a given championId and SummonerId
#e.g {'Rank':'unranked' , 'AVG(`MatchStatsTbl`.`MinionsKilled`)': 500}
def avgMinionsSummoner(champId,SummonerFk): 
    connection = create_connection()
    cursor =  connection.cursor()
    MinionsAvg = cursor.execute("SELECT `RankTbl`.`RankName` , AVG(`MatchStatsTbl`.`MinionsKilled`) FROM `MatchStatsTbl` JOIN `SummonerMatchTbl` on `MatchStatsTbl`.MatchStatsId = `SummonerMatchTbl`.`SummonerMatchId` JOIN `MatchTbl` on `SummonerMatchTbl`.`MatchFk` = `MatchTbl`.`MatchId` JOIN `RankTbl` on `RankTbl`.`RankId` = `MatchTbl`.`RankFk` WHERE `MatchTbl`.QueueType = 'CLASSIC' AND `SummonerMatchTbl`.`ChampionFk` = %s  and SummonerMatchTbl.SummonerFk = %s GROUP by `MatchTbl`.`RankFk`", (champId, int(SummonerFk)))
    MinionsAvg = cursor.fetchall()
    return MinionsAvg


#Gets average Damage Taken of champion from a given championId by rank
#e.g {'Rank':'unranked' , 'AVG(`MatchStatsTbl`.`DmgTaken`)': 500}
def avgDmgTakenAll():
    connection = create_connection()
    cursor =  connection.cursor()
    DmgTakenAvg = cursor.execute("SELECT `RankTbl`.`RankName` , AVG(`MatchStatsTbl`.`DmgTaken`) FROM `MatchStatsTbl` JOIN `SummonerMatchTbl` on `MatchStatsTbl`.MatchStatsId = `SummonerMatchTbl`.`SummonerMatchId` JOIN `MatchTbl` on `SummonerMatchTbl`.`MatchFk` = `MatchTbl`.`MatchId` JOIN `RankTbl` on `RankTbl`.`RankId` = `MatchTbl`.`RankFk` WHERE `MatchTbl`.QueueType = 'CLASSIC' GROUP by `MatchTbl`.`RankFk`")
    DmgTakenAvg = cursor.fetchall()
    return DmgTakenAvg

#Gets average Damage Taken of champion from a given championId by rank
#e.g {'Rank':'unranked' , 'AVG(`MatchStatsTbl`.`DmgTaken`)': 500}
def avgDmgTaken(champId):
    connection = create_connection()
    cursor =  connection.cursor()
    DmgTakenAvg = cursor.execute("SELECT `RankTbl`.`RankName` , AVG(`MatchStatsTbl`.`DmgTaken`) FROM `MatchStatsTbl` JOIN `SummonerMatchTbl` on `MatchStatsTbl`.MatchStatsId = `SummonerMatchTbl`.`SummonerMatchId` JOIN `MatchTbl` on `SummonerMatchTbl`.`MatchFk` = `MatchTbl`.`MatchId` JOIN `RankTbl` on `RankTbl`.`RankId` = `MatchTbl`.`RankFk` WHERE `MatchTbl`.QueueType = 'CLASSIC' AND `SummonerMatchTbl`.`ChampionFk` = %s GROUP by `MatchTbl`.`RankFk`",(int(champId),))
    DmgTakenAvg = cursor.fetchall()
    return DmgTakenAvg

#Gets average Damage Taken of champion from a given championId and SummonerId
#e.g {'Rank':'unranked' , 'AVG(`MatchStatsTbl`.`DmgTaken`)': 500}
def avgDmgTakenSummonerAll(SummonerFk): 
    connection = create_connection()
    cursor =  connection.cursor()
    DmgTakenAvg = cursor.execute("SELECT AVG(`MatchStatsTbl`.`DmgTaken`) FROM `MatchStatsTbl` JOIN `SummonerMatchTbl` on `MatchStatsTbl`.MatchStatsId = `SummonerMatchTbl`.`SummonerMatchId` JOIN `MatchTbl` on `SummonerMatchTbl`.`MatchFk` = `MatchTbl`.`MatchId` JOIN `RankTbl` on `RankTbl`.`RankId` = `MatchTbl`.`RankFk` WHERE `MatchTbl`.QueueType = 'CLASSIC' and SummonerMatchTbl.SummonerFk = %s", (int(SummonerFk),))
    DmgTakenAvg = cursor.fetchall()
    return DmgTakenAvg

#Gets average Damage Taken of champion from a given championId and SummonerId
#e.g {'Rank':'unranked' , 'AVG(`MatchStatsTbl`.`DmgTaken`)': 500}
def avgDmgTakenSummoner(champId,SummonerFk): 
    connection = create_connection()
    cursor =  connection.cursor()
    DmgTakenAvg = cursor.execute("SELECT `RankTbl`.`RankName` , AVG(`MatchStatsTbl`.`DmgTaken`) FROM `MatchStatsTbl` JOIN `SummonerMatchTbl` on `MatchStatsTbl`.MatchStatsId = `SummonerMatchTbl`.`SummonerMatchId` JOIN `MatchTbl` on `SummonerMatchTbl`.`MatchFk` = `MatchTbl`.`MatchId` JOIN `RankTbl` on `RankTbl`.`RankId` = `MatchTbl`.`RankFk` WHERE `MatchTbl`.QueueType = 'CLASSIC' AND `SummonerMatchTbl`.`ChampionFk` = %s and SummonerMatchTbl.SummonerFk = %s  GROUP by `MatchTbl`.`RankFk`", (champId, int(SummonerFk)))
    DmgTakenAvg = cursor.fetchall()
    return DmgTakenAvg

#Gets average Damage Dealt Across All ranks
#e.g {'Rank':'unranked' , 'AVG(`MatchStatsTbl`.`DmgDealt`)': 500}
def avgDmgDealtAll(): 
    connection = create_connection()
    cursor =  connection.cursor()
    DmgDealtAvg = cursor.execute("SELECT `RankTbl`.`RankName` , AVG(`MatchStatsTbl`.`DmgDealt`) FROM `MatchStatsTbl` JOIN `SummonerMatchTbl` on `MatchStatsTbl`.MatchStatsId = `SummonerMatchTbl`.`SummonerMatchId` JOIN `MatchTbl` on `SummonerMatchTbl`.`MatchFk` = `MatchTbl`.`MatchId` JOIN `RankTbl` on `RankTbl`.`RankId` = `MatchTbl`.`RankFk` WHERE `MatchTbl`.QueueType = 'CLASSIC' GROUP by `MatchTbl`.`RankFk`")
    DmgDealtAvg = cursor.fetchall()
    return DmgDealtAvg

#Gets average Damage Dealt of champion from a given championId by rank
#e.g {'Rank':'unranked' , 'AVG(`MatchStatsTbl`.`DmgDealt`)': 500}
def avgDmgDealt(champId): 
    connection = create_connection()
    cursor =  connection.cursor()
    DmgDealtAvg = cursor.execute("SELECT `RankTbl`.`RankName` , AVG(`MatchStatsTbl`.`DmgDealt`) FROM `MatchStatsTbl` JOIN `SummonerMatchTbl` on `MatchStatsTbl`.MatchStatsId = `SummonerMatchTbl`.`SummonerMatchId` JOIN `MatchTbl` on `SummonerMatchTbl`.`MatchFk` = `MatchTbl`.`MatchId` JOIN `RankTbl` on `RankTbl`.`RankId` = `MatchTbl`.`RankFk` WHERE `MatchTbl`.QueueType = 'CLASSIC' AND `SummonerMatchTbl`.`ChampionFk` = %s GROUP by `MatchTbl`.`RankFk`",(champId))
    DmgDealtAvg = cursor.fetchall()
    return DmgDealtAvg

#Gets average Damage Dealt of champion from a given championId by rank
#e.g {'Rank':'unranked' , 'AVG(`MatchStatsTbl`.`DmgDealt`)': 500}
def avgDmgDealtSummoner(champId,SummonerFk): 
    connection = create_connection()
    cursor =  connection.cursor()
    DmgDealtAvg = cursor.execute("SELECT `RankTbl`.`RankName` , AVG(`MatchStatsTbl`.`DmgDealt`) FROM `MatchStatsTbl` JOIN `SummonerMatchTbl` on `MatchStatsTbl`.MatchStatsId = `SummonerMatchTbl`.`SummonerMatchId` JOIN `MatchTbl` on `SummonerMatchTbl`.`MatchFk` = `MatchTbl`.`MatchId` JOIN `RankTbl` on `RankTbl`.`RankId` = `MatchTbl`.`RankFk` WHERE `MatchTbl`.QueueType = 'CLASSIC' AND `SummonerMatchTbl`.`ChampionFk` = %s  and SummonerMatchTbl.SummonerFk = %s  GROUP by `MatchTbl`.`RankFk`", (champId, int(SummonerFk)))
    DmgDealtAvg = cursor.fetchall()
    return DmgDealtAvg

#Gets average Damage Dealt of champion from a given SummonerId by rank
#e.g {'Rank':'unranked' , 'AVG(`MatchStatsTbl`.`DmgDealt`)': 500}
def avgDmgDealtSummonerAll(SummonerFk): 
    connection = create_connection()
    cursor =  connection.cursor()
    DmgDealtAvg = cursor.execute("SELECT AVG(`MatchStatsTbl`.`DmgDealt`) FROM `MatchStatsTbl` JOIN `SummonerMatchTbl` on `MatchStatsTbl`.MatchStatsId = `SummonerMatchTbl`.`SummonerMatchId` JOIN `MatchTbl` on `SummonerMatchTbl`.`MatchFk` = `MatchTbl`.`MatchId` JOIN `RankTbl` on `RankTbl`.`RankId` = `MatchTbl`.`RankFk` WHERE `MatchTbl`.QueueType = 'CLASSIC' and SummonerMatchTbl.SummonerFk = %s", (int(SummonerFk),))
    DmgDealtAvg = cursor.fetchall()
    return DmgDealtAvg


#Gets average gold across all ranks
#e.g {'Rank':'unranked' , 'AVG(`MatchStatsTbl`.`TotalGold`)': 500}
def avgGoldAll(): 
    connection = create_connection()
    cursor =  connection.cursor()
    TotalGoldAvg = cursor.execute("SELECT `RankTbl`.`RankName` , AVG(`MatchStatsTbl`.`TotalGold`) FROM `MatchStatsTbl` JOIN `SummonerMatchTbl` on `MatchStatsTbl`.MatchStatsId = `SummonerMatchTbl`.`SummonerMatchId` JOIN `MatchTbl` on `SummonerMatchTbl`.`MatchFk` = `MatchTbl`.`MatchId` JOIN `RankTbl` on `RankTbl`.`RankId` = `MatchTbl`.`RankFk` WHERE `MatchTbl`.QueueType = 'CLASSIC' GROUP by `MatchTbl`.`RankFk`")
    TotalGoldAvg = cursor.fetchall()
    return TotalGoldAvg

#Gets average gold of champion from a given championId by rank
#e.g {'Rank':'unranked' , 'AVG(`MatchStatsTbl`.`TotalGold`)': 500}
def avgGold(champId): 
    connection = create_connection()
    cursor =  connection.cursor()
    TotalGoldAvg = cursor.execute("SELECT `RankTbl`.`RankName` , AVG(`MatchStatsTbl`.`TotalGold`) FROM `MatchStatsTbl` JOIN `SummonerMatchTbl` on `MatchStatsTbl`.MatchStatsId = `SummonerMatchTbl`.`SummonerMatchId` JOIN `MatchTbl` on `SummonerMatchTbl`.`MatchFk` = `MatchTbl`.`MatchId` JOIN `RankTbl` on `RankTbl`.`RankId` = `MatchTbl`.`RankFk` WHERE `MatchTbl`.QueueType = 'CLASSIC' AND `SummonerMatchTbl`.`ChampionFk` = %s GROUP by `MatchTbl`.`RankFk`",(champId))
    TotalGoldAvg = cursor.fetchall()
    return TotalGoldAvg
  

#Gets average gold of champion from a given championId and SummonerIDd
#e.g {'Rank':'unranked' , 'AVG(`MatchStatsTbl`.`TotalGold`)': 500}
def avgGoldSummoner(champId,SummonerFk): 
    connection = create_connection()
    cursor =  connection.cursor()
    TotalGoldAvg = cursor.execute("SELECT `RankTbl`.`RankName` , AVG(`MatchStatsTbl`.`TotalGold`) FROM `MatchStatsTbl` JOIN `SummonerMatchTbl` on `MatchStatsTbl`.MatchStatsId = `SummonerMatchTbl`.`SummonerMatchId` JOIN `MatchTbl` on `SummonerMatchTbl`.`MatchFk` = `MatchTbl`.`MatchId` JOIN `RankTbl` on `RankTbl`.`RankId` = `MatchTbl`.`RankFk` WHERE `MatchTbl`.QueueType = 'CLASSIC' AND `SummonerMatchTbl`.`ChampionFk` = %s and SummonerMatchTbl.SummonerFk = %s  GROUP by `MatchTbl`.`RankFk`", (champId, int(SummonerFk)))
    TotalGoldAvg = cursor.fetchall()
    return TotalGoldAvg

#Gets average gold of champion from a given championId and SummonerIDd
#e.g {'Rank':'unranked' , 'AVG(`MatchStatsTbl`.`TotalGold`)': 500}
def avgGoldSummonerAll(SummonerFk): 
    connection = create_connection()
    cursor =  connection.cursor()
    TotalGoldAvg = cursor.execute("SELECT AVG(`MatchStatsTbl`.`TotalGold`) FROM `MatchStatsTbl` JOIN `SummonerMatchTbl` on `MatchStatsTbl`.MatchStatsId = `SummonerMatchTbl`.`SummonerMatchId` JOIN `MatchTbl` on `SummonerMatchTbl`.`MatchFk` = `MatchTbl`.`MatchId` JOIN `RankTbl` on `RankTbl`.`RankId` = `MatchTbl`.`RankFk` WHERE `MatchTbl`.QueueType = 'CLASSIC' AND SummonerMatchTbl.SummonerFk = %s", (int(SummonerFk),))
    TotalGoldAvg = cursor.fetchall()
    return TotalGoldAvg

#Gets Best (Most Wins) Items
#Passes value to getItemDescriptions
def commonItems(champId): 
    connection = create_connection()
    cursor =  connection.cursor()
    commonItems = cursor.execute("SELECT item1, COUNT(item1) ,item2 , COUNT(item2) ,item3 , COUNT(item3) ,item4 , COUNT(item4),item5 , COUNT(item5),item6 , COUNT(item6) FROM MatchStatsTbl JOIN SummonerMatchTbl on SummonerMatchFk = SummonerMatchTbl.SummonerMatchId WHERE SummonerMatchTbl.ChampionFk = %s",(champId))
    commonItems = cursor.fetchone()
    
    commonItems = getItemDescriptions(commonItems)
    return commonItems

#Gets Best (Most Wins) Items
#Passes value to getItemDescriptions
def bestItems(champId): 
    connection = create_connection()
    cursor =  connection.cursor()
    bestItems = cursor.execute("SELECT item1, COUNT(item1) ,item2 , COUNT(item2) ,item3 , COUNT(item3) ,item4 , COUNT(item4),item5 , COUNT(item5),item6 , COUNT(item6) FROM MatchStatsTbl JOIN SummonerMatchTbl on SummonerMatchFk = SummonerMatchTbl.SummonerMatchId WHERE SummonerMatchTbl.ChampionFk = %s AND win = 1 GROUP BY item2 ORDER BY `COUNT(item2)` DESC LIMIT 1 ",(champId))
    bestItems = cursor.fetchone()
    from championsRequest import getItemDescriptions
    bestItems = getItemDescriptions(bestItems)
    return bestItems

#Gets Common (Most Occurences) Primary Runes
#Passes value to runeImagesFromDatabase
def commonRunes(champId): 
    connection = create_connection()
    cursor =  connection.cursor()
    Runes = cursor.execute("SELECT PrimaryKeyStone, COUNT(PrimaryKeyStone), PrimarySlot1 , COUNT(PrimarySlot1) ,PrimarySlot2 , COUNT(PrimarySlot2) ,PrimarySlot3 , COUNT(PrimarySlot3) FROM MatchStatsTbl JOIN SummonerMatchTbl on SummonerMatchFk = SummonerMatchTbl.SummonerMatchId WHERE SummonerMatchTbl.ChampionFk = %s GROUP BY PrimaryKeyStone ORDER BY PrimaryKeyStone DESC LIMIT 1 ",(champId))
    Runes = cursor.fetchone()
    runeList = runeImagesFromDatabase(Runes)
    return runeList

#Gets Best (Most Wins) Primary runes
#Passes value to runeImagesFromDatabase
def bestRunes(champId): 
    connection = create_connection()
    cursor =  connection.cursor()
    bestRunes = cursor.execute("SELECT PrimaryKeyStone, COUNT(PrimaryKeyStone), PrimarySlot1 , COUNT(PrimarySlot1) ,PrimarySlot2 , COUNT(PrimarySlot2) ,PrimarySlot3 , COUNT(PrimarySlot3) FROM MatchStatsTbl JOIN SummonerMatchTbl on SummonerMatchFk = SummonerMatchTbl.SummonerMatchId AND Win = 1 WHERE SummonerMatchTbl.ChampionFk = %s GROUP BY PrimaryKeyStone ORDER BY PrimaryKeyStone DESC LIMIT 1",(champId))
    bestRunes = cursor.fetchone()
    bestRuneList = runeImagesFromDatabase(bestRunes)
    return bestRuneList

#Gets Common (Most Occurences) secondary runes
#Passes value to runeImagesFromDatabase
def commonSecondaryRunes(champId): 
    connection = create_connection()
    cursor =  connection.cursor()
    SecondRunes =  cursor.execute("SELECT SecondarySlot1, COUNT(SecondarySlot1), SecondarySlot2 , COUNT(SecondarySlot2) FROM MatchStatsTbl JOIN SummonerMatchTbl on SummonerMatchFk = SummonerMatchTbl.SummonerMatchId WHERE SummonerMatchTbl.ChampionFk = %s GROUP BY SecondarySlot1 ORDER BY `COUNT(SecondarySlot1)` DESC LIMIT 1",(champId))
    SecondRunes = cursor.fetchone()
    SecondRunes = runeImagesFromDatabase(SecondRunes)
    return SecondRunes

#Gets Best (Most Wins) secondary runes
#Passes value to runeImagesFromDatabase
def bestSecondaryRunes(champId): 
    connection = create_connection()
    cursor =  connection.cursor()
    bestSecondRunes = cursor.execute("SELECT SecondarySlot1, COUNT(SecondarySlot1), SecondarySlot2 , COUNT(SecondarySlot2) FROM MatchStatsTbl JOIN SummonerMatchTbl on SummonerMatchFk = SummonerMatchTbl.SummonerMatchId WHERE SummonerMatchTbl.ChampionFk = %s AND Win = 1 GROUP BY SecondarySlot1 ORDER BY `COUNT(SecondarySlot1)` DESC LIMIT 1",(champId))
    bestSecondRunes = cursor.fetchone()
    bestSecondRunes = runeImagesFromDatabase(bestSecondRunes)
    return bestSecondRunes

#Pass runes from database into a dictionary containing RuneName,RuneLink and Description 
#Returns a List of Dictionary of runes
def runeImagesFromDatabase(runes): 
    RuneList = []
    tempRunes = []
    for key in runes:
        if "COUNT" in key:
            count = runes[key]
        else:
            tempRunes.append(runes[key])
    MainRune = None
    runes = list(runes)
    runes = list(filter(lambda a: a != count, runes))
    for val in tempRunes:
            rune,MainRune = getRunesImages(val)
            RuneList.append(rune)
            MainRune = MainRune
    RuneList.insert(0, MainRune)

    return RuneList

#Gets the KDA of a given champion by champId
#e.g {' AVG(kills)':5 , 'AVG(deaths)': 7,AVG(assists): 2}
#Returns a String 5/7/2
def kdaFromDatabase(champId): 
    connection = create_connection()
    cursor =  connection.cursor()
    kda = cursor.execute("SELECT AVG(kills), AVG(deaths), AVG(assists) FROM MatchStatsTbl JOIN SummonerMatchTbl on SummonerMatchFk = SummonerMatchTbl.SummonerMatchId WHERE SummonerMatchTbl.ChampionFk = %s",(champId))
    kda = cursor.fetchone()
    kda = str(int(kda['AVG(kills)'])) + "/" + str(int(kda['AVG(deaths)'])) + "/" + str(int(kda['AVG(assists)']))
    return kda

#Gets the KDA of a given champion by champId and SummonerId
#e.g {' AVG(kills)':5 , 'AVG(deaths)': 7,AVG(assists): 2}
#Returns a String 5/7/2
def kdaFromDatabaseSummoner(champId,SummonerFk): 
    connection = create_connection()
    cursor =  connection.cursor()
    kda = cursor.execute("SELECT AVG(kills), AVG(deaths), AVG(assists) FROM MatchStatsTbl JOIN SummonerMatchTbl on SummonerMatchFk = SummonerMatchTbl.SummonerMatchId WHERE SummonerMatchTbl.ChampionFk = %s and SummonerMatchTbl.SummonerFk = %s", (champId, int(SummonerFk)))
    kda = cursor.fetchall()
    kda = str(kda[0]['AVG(kills)']) + "/" + str(kda[0]['AVG(deaths)']) + "/" + str(kda[0]['AVG(assists)'])
    return kda

#Gets the most frequently played lane of a given champion by champId
def laneFromDatabase(champId): 
    connection = create_connection()
    cursor =  connection.cursor()
    position = cursor.execute("SELECT Lane, COUNT(Lane) FROM MatchStatsTbl JOIN SummonerMatchTbl on SummonerMatchFk = SummonerMatchTbl.SummonerMatchId WHERE SummonerMatchTbl.ChampionFk = %s", (champId))
    position = cursor.fetchone()
    position = position['Lane']
    return position

#Gets most played Position of champion from a given championId and SummonerId
#e.g {'Lane':'MIDDLE' , 'COUNT(Lane)': 500}
def laneFromDatabaseSummoner(champId,SummonerFk): 
    connection = create_connection()
    cursor =  connection.cursor()
    position = cursor.execute("SELECT Lane, COUNT(Lane) FROM MatchStatsTbl JOIN SummonerMatchTbl on SummonerMatchFk = SummonerMatchTbl.SummonerMatchId WHERE SummonerMatchTbl.ChampionFk = %s and SummonerMatchTbl.SummonerFk = %s GROUP BY Lane ORDER BY PrimaryKeyStone DESC ", (champId, int(SummonerFk)))
    position = cursor.fetchone()
    position = position['Lane']
    return position

#Gets the SummonerId from a given SummonerName
def getSummonerIdFromDatabase(SummonerName): 
    connection = create_connection()
    cursor =  connection.cursor()
    sql_query = "SELECT SummonerUserTbl.SummonerID FROM SummonerUserTbl WHERE SummonerName = %s"
    # Execute the SQL query with the parameter
    cursor.execute(sql_query, (SummonerName,))
    result = cursor.fetchone()

    if result:
        summoner_id = result[0]
        print(f"Summoner ID for {SummonerName}: {summoner_id}")
    else:
        print(f"No summoner found with the name {SummonerName}")
    return summoner_id

#Gets ItemLink
def getItemLink(id): 
    connection = create_connection()
    cursor =  connection.cursor()
    Link = cursor.execute("SELECT `ItemLink` FROM `ItemTbl` WHERE ItemID = %s", (id))
    Link = cursor.fetchone()
    Link = Link['ItemLink']
    return Link

#Gets All Champions
def getAllChampions(): 
    connection = create_connection()
    cursor =  connection.cursor()
    champ = cursor.execute("SELECT * FROM `ChampionTbl`")
    champ = cursor.fetchall() #######
    championDict = []
    for m in champ:
        print(m[0])
        Champion = {}
        Champion['ChampionId'] = m[0]
        Champion['ChampionName'] = m[1]
        championDict.append(Champion)

    return championDict

#Gets the Top players by wins from database
def getBestPlayers():
    connection = create_connection()
    cursor =  connection.cursor()
    players = cursor.execute("SELECT DISTINCT SummonerName, COUNT(MatchStatsTbl.Win), AVG(MatchStatsTbl.kills),AVG(MatchStatsTbl.assists), AVG(MatchStatsTbl.deaths), AVG(MatchStatsTbl.BaronKills), AVG(MatchStatsTbl.DragonKills) FROM `SummonerUserTbl` JOIN SummonerMatchTbl on SummonerID = SummonerMatchTbl.SummonerFk JOIN MatchStatsTbl on SummonerMatchTbl.SummonerMatchId = MatchStatsTbl.SummonerMatchFk WHERE MatchStatsTbl.Win = 1 GROUP BY SummonerName ORDER by COUNT(MatchStatsTbl.Win) DESC LIMIT 25")
    players = cursor.fetchall()
    return players

#Gets Champion table
def getChampionAverages():
    connection = create_connection()
    cursor =  connection.cursor()
    query = ('SELECT `ChampionTbl`.`ChampionName`, AVG(`MatchStatsTbl`.`kills`),AVG(`MatchStatsTbl`.`deaths`),AVG(`MatchStatsTbl`.`assists`), AVG(`MatchStatsTbl`.`Win`), AVG(`MatchTbl`.`GameDuration`) FROM `SummonerMatchTbl` JOIN `MatchStatsTbl` ON `MatchStatsTbl`.SummonerMatchFk = `SummonerMatchTbl`.SummonerMatchId   JOIN `MatchTbl` ON `MatchTbl`.`MatchId` = `SummonerMatchTbl`.`MatchFk`  JOIN `ChampionTbl` ON  `SummonerMatchTbl`.`ChampionFk` = `ChampionTbl`.`ChampionId`   WHERE `MatchTbl`.`QueueType` = "CLASSIC"  GROUP BY `ChampionTbl`.`ChampionId`')
    cursor.execute(query)
    data = cursor.fetchall()
    return data

#Gets the best player from a given Champion
def getChampionBestPlayers(ChampId):
    connection = create_connection()
    cursor =  connection.cursor()
    query = ("SELECT DISTINCT SummonerName, COUNT(MatchStatsTbl.Win), AVG(MatchStatsTbl.kills),AVG(MatchStatsTbl.assists), AVG(MatchStatsTbl.deaths), AVG(MatchStatsTbl.BaronKills), AVG(MatchStatsTbl.DragonKills) FROM `SummonerUserTbl` JOIN SummonerMatchTbl on SummonerID = SummonerMatchTbl.SummonerFk JOIN MatchStatsTbl on SummonerMatchTbl.SummonerMatchId = MatchStatsTbl.SummonerMatchFk WHERE MatchStatsTbl.Win = 1 and SummonerMatchTbl.ChampionFk = %s GROUP BY SummonerName ORDER by COUNT(MatchStatsTbl.Win) DESC LIMIT 15 ")
    cursor.execute(query, (ChampId,))
    data = cursor.fetchall()
    return data

#Insert user into database
def insertUser(SummonerName):
    connection = create_connection()
    cursor =  connection.cursor()
    cursor.execute("INSERT INTO `SummonerUserTbl`(`SummonerName`) VALUES (%s )", (SummonerName,))
    connection.commit()

    SummonerFk = cursor.execute("SELECT SummonerUserTbl.SummonerID from SummonerUserTbl where SummonerName = %s", (SummonerName, ))
    SummonerFk = cursor.fetchall()
    try:
        SummonerFk = SummonerFk[0]['SummonerID']
    except:
        SummonerFk = None
    return SummonerFk    

def getRankId(Rank):
    if not Rank:  # Check if the array is empty
        return 0  # or whatever default value you want to return in this case
    
    for rank in Rank:
        if rank['queueType'] == "RANKED_SOLO_5x5":
            userRank = rank['tier']
            print("USER RANK ====", userRank)
    
    print("USER RANK ====", userRank)
    connection = create_connection()
    cursor = connection.cursor()
    cursor.execute("SELECT `RankId` FROM `RankTbl` WHERE `RankName` = (%s)", (userRank,))
    RankId = cursor.fetchone()
    if RankId:
        RankId = RankId[0]
    else:
        RankId = 0  # or whatever default value you want to return if the rank is not found
    cursor.close()
    connection.close()
    return RankId


#Check if a match exists
def matchCheck(matchId):
    connection = create_connection()
    cursor =  connection.cursor()
    cursor.execute("SELECT `MatchId` FROM `MatchTbl` WHERE `MatchId` = (%s)", (str(matchId) ,))
    matchCheck = cursor.fetchone()
    if matchCheck != None:
        matchCheck = matchCheck[0]
    else:
        pass
    return matchCheck
#Inserts a match into the database
def insertMatch(matchId,Patch,GameType,RankId,GameDuration):
    connection = create_connection()
    cursor =  connection.cursor()
    print(matchId)
    print(Patch)
    print(GameType)
    print(RankId)
    cursor.execute("INSERT INTO `MatchTbl`(`MatchId`, `Patch`,  `QueueType`, `RankFk`,`GameDuration`) VALUES (%s ,%s , %s , %s, %s)", (matchId,Patch,GameType,int(RankId),int(GameDuration)))
    connection.commit()
#Inserts the match stats of a user into the database
def insertMatchStats(SummMatchId ,cs,dmgDealt,dmgTaken,TurretDmgDealt,goldEarned,Role,win,Item1,Item2,Item3,Item4,Item5,Item6,kills,deaths,asssts,PK1,PK2,PK3,PK4,SK1,SK2,spell1,spell2,masteryPoints,Enemy,dragonKills,baronKills):
    connection = create_connection()
    cursor =  connection.cursor()
    cursor.execute("INSERT INTO `MatchStatsTbl`(`SummonerMatchFk`, `MinionsKilled`, `DmgDealt`, `DmgTaken`, `TurretDmgDealt`, `TotalGold`, `Lane`, `Win`, `item1`, `item2`, `item3`, `item4`, `item5`, `item6`, `kills`, `deaths`, `assists`, `PrimaryKeyStone`, `PrimarySlot1`, `PrimarySlot2`, `PrimarySlot3`, `SecondarySlot1`, `SecondarySlot2`, `SummonerSpell1`, `SummonerSpell2`, `CurrentMasteryPoints`, `EnemyChampionFk`, `DragonKills`, `BaronKills`) VALUES (%s, %s , %s ,%s , %s , %s , %s , %s , %s , %s , %s , %s , %s , %s , %s , %s , %s , %s , %s , %s , %s , %s , %s , %s , %s , %s , %s , %s , %s)" ,(str(SummMatchId) ,cs,dmgDealt,dmgTaken,TurretDmgDealt,goldEarned,Role,win,Item1,Item2,Item3,Item4,Item5,Item6,kills,deaths,asssts,PK1,PK2,PK3,PK4,SK1,SK2,spell1,spell2,masteryPoints,Enemy,dragonKills,baronKills))
    connection.commit()
#Inserts a match into the SummonerMatchTbl
def insertSummMatch(SummonerId,MatchVerify,Champion):
    connection = create_connection()
    cursor =  connection.cursor()
    cursor.execute("INSERT INTO `SummonerMatchTbl`(`SummonerFk`, `MatchFk`, `ChampionFk`) VALUES (%s , %s , %s)", (SummonerId,MatchVerify,Champion))
    connection.commit()
    cursor.execute("SELECT `SummonerMatchId` FROM `SummonerMatchTbl` WHERE `MatchFk` = (%s) AND `SummonerFk` = (%s)", (str(MatchVerify) ,SummonerId))    
    SummMatchId = cursor.fetchone()        
    SummMatchId = SummMatchId['SummonerMatchId']
    return SummMatchId

#Gets the id of a champion from a string championName
def getChampId(champion):
    connection = create_connection()
    cursor =  connection.cursor()
    print(champion)
    cursor.execute("SELECT `ChampionId` FROM `ChampionTbl` WHERE `ChampionName` = (%s)", (champion, ))
    champion = cursor.fetchone()
    champion = champion[0]
    print(champion)
    return champion

def Normalise(stri):
    stri = str(stri)
    stri = stri.replace('[', '')
    stri = stri.replace(']', '')
    stri = stri.replace("'", '')
    stri = stri.replace('(', '')
    stri = stri.replace(')', '')
    stri = stri.replace(",", '')
    return stri

#Check if the summoner has played in a match stored in database.
def checkSummMatch(SummonerId,MatchId):
    connection = create_connection()
    cursor =  connection.cursor()
    cursor.execute("SELECT `SummonerMatchId` FROM `SummonerMatchTbl` WHERE `MatchFk` = (%s) AND `SummonerFk` = (%s)", (str(MatchId) ,SummonerId))    
    SummMatchId = cursor.fetchone()
    cursor.close()
    connection.close()

    print("SummonerMatchId")
    print(SummMatchId)
    if SummMatchId is not None:
        SummMatchId = SummMatchId[0]
        print("SummonerMatchId:", SummMatchId)
    else:
        pass
    
    return SummMatchId

#Gets the championName from champId
def getChampName(champion):
    connection = create_connection()
    cursor =  connection.cursor()
    cursor.execute("SELECT `ChampionName` FROM `ChampionTbl` WHERE `ChampionId` = (%s)", (champion, ))
    champion = cursor.fetchone()
    champion = champion['ChampionName']
    return champion

#Gets Count of all games played
def getAllGamesCount():
    connection = create_connection()
    cursor =  connection.cursor()
    cursor.execute("SELECT COUNT(`MatchStatsTbl`.Win) FROM `MatchStatsTbl`")
    games = cursor.fetchall()
    return games

def avgDmgDealtSummonerAll(SummonerFk): 
    connection = create_connection()
    cursor =  connection.cursor()
    DmgDealtAvg = cursor.execute("SELECT AVG(`MatchStatsTbl`.`DmgDealt`) FROM `MatchStatsTbl` JOIN `SummonerMatchTbl` on `MatchStatsTbl`.MatchStatsId = `SummonerMatchTbl`.`SummonerMatchId` JOIN `MatchTbl` on `SummonerMatchTbl`.`MatchFk` = `MatchTbl`.`MatchId` JOIN `RankTbl` on `RankTbl`.`RankId` = `MatchTbl`.`RankFk` WHERE `MatchTbl`.QueueType = 'CLASSIC' and SummonerMatchTbl.SummonerFk = %s", (int(SummonerFk),))
    DmgDealtAvg = cursor.fetchall()
    return DmgDealtAvg

#Gets average Dragon across all ranks
#e.g {'Rank':'unranked' , 'AVG(`MatchStatsTbl`.`TotalGold`)': 500}
def avgDragonAll(): 
    connection = create_connection()
    cursor =  connection.cursor()
    TotalDragonAvg = cursor.execute("SELECT `RankTbl`.`RankName` , AVG(`MatchStatsTbl`.`DragonKills`) FROM `MatchStatsTbl` JOIN `SummonerMatchTbl` on `MatchStatsTbl`.MatchStatsId = `SummonerMatchTbl`.`SummonerMatchId` JOIN `MatchTbl` on `SummonerMatchTbl`.`MatchFk` = `MatchTbl`.`MatchId` JOIN `RankTbl` on `RankTbl`.`RankId` = `MatchTbl`.`RankFk` WHERE `MatchTbl`.QueueType = 'CLASSIC' GROUP by `MatchTbl`.`RankFk`")
    TotalDragonAvg = cursor.fetchall()
    return TotalDragonAvg

def avgDragonSummoner(SummonerFk): 
    connection = create_connection()
    cursor =  connection.cursor()
    TotalDragonAvg = cursor.execute("SELECT AVG(`MatchStatsTbl`.`DragonKills`) FROM `MatchStatsTbl` JOIN `SummonerMatchTbl` on `MatchStatsTbl`.MatchStatsId = `SummonerMatchTbl`.`SummonerMatchId` JOIN `MatchTbl` on `SummonerMatchTbl`.`MatchFk` = `MatchTbl`.`MatchId` JOIN `RankTbl` on `RankTbl`.`RankId` = `MatchTbl`.`RankFk` WHERE `MatchTbl`.QueueType = 'CLASSIC' AND SummonerMatchTbl.SummonerFk = %s", (int(SummonerFk),))
    TotalDragonAvg = cursor.fetchall()
    print(TotalDragonAvg)
    return TotalDragonAvg

def avgBaronAll(): 
    connection = create_connection()
    cursor =  connection.cursor()
    TotalDragonAvg = cursor.execute("SELECT `RankTbl`.`RankName` , AVG(`MatchStatsTbl`.`BaronKills`) FROM `MatchStatsTbl` JOIN `SummonerMatchTbl` on `MatchStatsTbl`.MatchStatsId = `SummonerMatchTbl`.`SummonerMatchId` JOIN `MatchTbl` on `SummonerMatchTbl`.`MatchFk` = `MatchTbl`.`MatchId` JOIN `RankTbl` on `RankTbl`.`RankId` = `MatchTbl`.`RankFk` WHERE `MatchTbl`.QueueType = 'CLASSIC' GROUP by `MatchTbl`.`RankFk`")
    TotalDragonAvg = cursor.fetchall()
    return TotalDragonAvg

def avgBaronSummoner(SummonerFk): 
    connection = create_connection()
    cursor =  connection.cursor()
    TotalDragonAvg = cursor.execute("SELECT AVG(`MatchStatsTbl`.`BaronKills`) FROM `MatchStatsTbl` JOIN `SummonerMatchTbl` on `MatchStatsTbl`.MatchStatsId = `SummonerMatchTbl`.`SummonerMatchId` JOIN `MatchTbl` on `SummonerMatchTbl`.`MatchFk` = `MatchTbl`.`MatchId` JOIN `RankTbl` on `RankTbl`.`RankId` = `MatchTbl`.`RankFk` WHERE `MatchTbl`.QueueType = 'CLASSIC' AND SummonerMatchTbl.SummonerFk = %s", (int(SummonerFk),))
    TotalDragonAvg = cursor.fetchall()
    print(TotalDragonAvg)
    return TotalDragonAvg