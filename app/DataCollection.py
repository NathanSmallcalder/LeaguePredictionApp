import sys
from RiotApiCalls import *
from championsRequest import *
from databaseQuries import *
import mysql.connector
import json
from config import *
import time
import requests

def Normalise(stri):
    stri = str(stri)
    stri = stri.replace('[', '')
    stri = stri.replace(']', '')
    stri = stri.replace("'", '')
    stri = stri.replace('(', '')
    stri = stri.replace(')', '')
    stri = stri.replace(",", '')
    return stri

config = {
    'user': 'league_user',
    'password': 'StrongPassword123!',
    'host': 'localhost',
    'port': 3306,
    'database': 'LeagueStats',
    'buffered': True,
    'auth_plugin': 'mysql_native_password'
}

connection = mysql.connector.connect(**config)
cursor = connection.cursor(buffered=True)

RegionStart = "europe"
Region = "EUW1"

def summoner(summonerName, tagline):
    try:
        print("called")
        connection.autocommit = True
        Summoner = getPuuid(RegionStart, summonerName, tagline)
        puuid = Summoner['puuid']
        SummonerInfo = getSummonerDetails(Region, puuid)
        print(SummonerInfo)
        SummId = SummonerInfo['puuid']
        RankedDetails = getRankedStats(Region, SummId)
        mastery = getMasteryStats(Region, puuid)
        print(mastery)
        cursor.execute("INSERT INTO `SummonerUserTbl`(`SummonerName`) VALUES (%s )", (summonerName,))
        connection.commit()

        MatchIDs = requests.get(
            "https://europe.api.riotgames.com/lol/match/v5/matches/by-puuid/"
            + SummonerInfo['puuid']
            + "/ids?start=0&count=20&api_key="
            + api_key
        )
        MatchIDs = MatchIDs.json()
        matchData = getMatches("euw1", Summoner, MatchIDs, SummonerInfo, RankedDetails, mastery)
        matchData2 = getsMatchData()
        Match = matchData2[1]['MatchIDS']
        Match = Normalise(Match)

        i = 0
        for MatchId in MatchIDs:
                MatchData = requests.get(
                    "https://europe.api.riotgames.com/lol/match/v5/matches/" + MatchId + "?api_key=" + api_key
                ).json()

                print("https://europe.api.riotgames.com/lol/match/v5/matches/" + MatchId + "?api_key=" + api_key)

                ii = 0
                champList = []
                while ii < 10:
                    champion = MatchData['info']['participants'][ii]['championName']
                    cursor.execute("SELECT `ChampionId` FROM `ChampionTbl` WHERE `ChampionName` = (%s)", (champion,))
                    Champion = cursor.fetchall()
                    Champion = Normalise(Champion)
                    print("Champion", Champion, champion)
                    champList.append(Champion)
                    ii += 1

                print(champList)
                # Blue team
                BaronKillsBlue = int(MatchData['info']['teams'][0]['objectives']['baron']['kills'])
                ChampionKillsBlue = int(MatchData['info']['teams'][0]['objectives']['champion']['kills'])
                DragonKillsBlue = int(MatchData['info']['teams'][0]['objectives']['dragon']['kills'])
                BlueRiftKills = int(MatchData['info']['teams'][0]['objectives']['riftHerald']['kills'])
                towerKillsBlue = int(MatchData['info']['teams'][0]['objectives']['tower']['kills'])
                BlueWin = int(MatchData['info']['teams'][0]['win'])

                # Red team
                BaronKillsRed = int(MatchData['info']['teams'][1]['objectives']['baron']['kills'])
                ChampionKillsRed = int(MatchData['info']['teams'][1]['objectives']['champion']['kills'])
                DragonKillsRed = int(MatchData['info']['teams'][1]['objectives']['dragon']['kills'])
                RedRiftKills = int(MatchData['info']['teams'][1]['objectives']['riftHerald']['kills'])
                towerKillsRed = int(MatchData['info']['teams'][1]['objectives']['tower']['kills'])
                RedWin = int(MatchData['info']['teams'][1]['win'])

                cursor.execute("SELECT `MatchFk` FROM `TeamMatchTbl` WHERE `MatchFk` = (%s)", (str(MatchId),))
                matchCheck = cursor.fetchone()
                Match = matchData2[i]['MatchIDS']
                Patch = matchData2[i]['gameVersion']
                Rank = matchData2[i]['Rank']
                GameType = matchData2[i]['GameType']
                GameDuration = matchData[i]['GameDuration']

                Match = Normalise(Match)
                GameType = Normalise(GameType)
                Rank = Normalise(Rank)
                Patch = Normalise(Patch)

                cursor.execute("SELECT `RankId` FROM `RankTbl` WHERE `RankName` = (%s)", (Rank,))
                RankId = cursor.fetchone()
                RankId = Rank[0]

                print(matchCheck)
                if matchCheck == None:
                    insertMatch(MatchId,Patch,GameType,RankId,GameDuration)

                    #PlayerMatchData
                    cursor.execute("SELECT `SummonerID` FROM `SummonerUserTbl` WHERE `SummonerName` = (%s)", (summonerName ,))
                    SummonerID = cursor.fetchone()
                    SummonerID = int(Normalise(SummonerID))
                    print(SummonerID)
                    champion = matchData[i]['champion']

                    cursor.execute("SELECT `ChampionId` FROM `ChampionTbl` WHERE `ChampionName` = (%s)", (champion, ))
                    Champion = cursor.fetchall()
                    Champion = Normalise(Champion)
                    print("Champion", Champion)
                    
                    cursor.execute("SELECT `MatchId` FROM `MatchTbl` WHERE `MatchId` = (%s)", (str(Match) ,))
                    MatchVerify = cursor.fetchone()
                    MatchVerify = Normalise(MatchVerify)
                    print("MatchPlayed", MatchVerify)

                    win = matchData[i]['win']

                    masteryScore = 0
                    
                    for m in mastery:
                        if int(Champion)== int(float(m['championPoints'])):
                            masteryScore = int(float(m['championPoints']))
                            print(masteryScore)
                            break

                    masteryPoints = getSingleMasteryScore(Champion,mastery)
                    
                    cs = matchData[i]['cs']
                    dmgDealt = matchData[i]['physicalDamageDealtToChampions']
                    dmgTaken = matchData[i]['physicalDamageTaken']
                    spell1 = matchData[i]['SummonerSpell1']
                    spell2 = matchData[i]['SummonerSpell2']
                    print(spell1, spell2)
                    TurretDmgDealt = matchData[i]['TowerDamageDealt']
                    goldEarned = matchData[i]['goldEarned']
                    Role= matchData[i]['Role']
                
                    #CHANGE BOTTOM TO SUPPORT IF SUPPORT
                    if int(Champion) == 412 or 350 or 117 or 235 or 497 or 111 or 99 or 267 or 43 or 53 or 555 or 25 or 1 or 22 or 16 or 89 or 101 or 12 or 143 or 40 or 147 or 37 or 26 or 888 or 50 or 432 or 32 or 63 or 74 or 201 or 29 or 161 or 44 or 526 or 57 or 518:
                        if Role == "BOTTOM" and cs < 75:
                            Role = "SUPPORT"

                    Item1 = matchData[i]['Items'][0]
                    Item2 = matchData[i]['Items'][1]
                    Item3 = matchData[i]['Items'][2]
                    Item4 = matchData[i]['Items'][3]
                    Item5 = matchData[i]['Items'][4]
                    Item6 = matchData[i]['Items'][5]

                    print(Item1, "Item 1 " , Item2,Item3, " " , Item4, " " ,Item5, " " , Item6)


                    dragonKills = matchData[i]['dragonKills']
                    baronKills = matchData[i]['baronKills']
                    
                    kills = matchData[i]['kills']
                    deaths = matchData[i]['deaths']
                    assists = matchData[i]['assists']

                    PK1 = matchData[i]['PrimaryKeyStone'][0]

                    PK2 = matchData[i]['PrimaryKeyStone'][1]
                    PK3 = matchData[i]['PrimaryKeyStone'][2]
                    PK4 = matchData[i]['PrimaryKeyStone'][3]
                    SK1 = matchData[i]['SecondaryKeyStone'][0]
                    SK2 = matchData[i]['SecondaryKeyStone'][1]
                    visionScore = matchData[i]['visionScore']
                    EmemyLane = matchData[i]['EnemyChamp']

                    cursor.execute("SELECT `ChampionId` FROM `ChampionTbl` WHERE `ChampionName` = (%s)", (EmemyLane, ))
                    Enemy = cursor.fetchone()
                    Enemy = Normalise(Enemy)
                    
                    if(Enemy == "None"):
                        Enemy = 0
                    if MatchVerify == "None":
                        cursor.execute("INSERT INTO `MatchTbl`(`MatchId`, `Patch`,  `QueueType`, `RankFk`,`GameDuration`) VALUES (%s ,%s , %s , %s, %s)", (Match,Patch,GameType,int(RankId),int(GameDuration)))
                        connection.commit()
                        cursor.execute("SELECT `MatchId` FROM `MatchTbl` WHERE `MatchId` = (%s)", (str(Match) ,))
                        MatchVerify = cursor.fetchone()
                        MatchVerify = Normalise(MatchVerify)
                        print("Mattch Inserted", MatchVerify)
                
                    else:
                        print("Pass")
                        pass

                    print("SUMMONER ID = ", SummonerID, "MATCH = ", MatchVerify, "CHAMPION = ", Champion)
                    cursor.execute("INSERT INTO `SummonerMatchTbl`(`SummonerFk`, `MatchFk`, `ChampionFk`) VALUES (%s , %s , %s)", (SummonerID, MatchVerify, Champion))
                    connection.commit()
                
                    # Retrieve the SummonerMatchId after insertion
                    cursor.execute("SELECT `SummonerMatchId` FROM `SummonerMatchTbl` WHERE `MatchFk` = %s AND `SummonerFk` = %s", (MatchVerify, SummonerID))    
                    SummMatchId = cursor.fetchone()        
                    if SummMatchId:
                        SummMatchId = SummMatchId[0]  # Extracting the value from the tuple

                    print("SummMatchID = ", SummMatchId)
                    query = """INSERT INTO `MatchStatsTbl`
                (`SummonerMatchFk`, `MinionsKilled`, `DmgDealt`, `DmgTaken`, `TurretDmgDealt`,
                `TotalGold`, `Lane`, `Win`, `item1`, `item2`, `item3`, `item4`, `item5`, `item6`,
                `kills`, `deaths`, `assists`, `PrimaryKeyStone`, `PrimarySlot1`, `PrimarySlot2`,
                `PrimarySlot3`, `SecondarySlot1`, `SecondarySlot2`, `SummonerSpell1`,
                `SummonerSpell2`, `CurrentMasteryPoints`, `EnemyChampionFk`, `DragonKills`, `BaronKills`)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"""
                    values = (SummMatchId, cs, dmgDealt, dmgTaken, TurretDmgDealt, goldEarned, Role, win,
                        Item1, Item2, Item3, Item4, Item5, Item6, kills, deaths, assists, PK1, PK2,
                        PK3, PK4, SK1, SK2, spell1, spell2, masteryPoints, Enemy, dragonKills, baronKills,visionScore)

                
                    # Assuming the following variables are defined: cs, dmgDealt, dmgTaken, TurretDmgDealt, goldEarned, Role, win, Item1, Item2, Item3, Item4, Item5, Item6, kills, deaths, assists, PK1, PK2, PK3, PK4, SK1, SK2, spell1, spell2, masteryPoints, Enemy, dragonKills, baronKills
                    cursor.execute("INSERT INTO `MatchStatsTbl`(`SummonerMatchFk`, `MinionsKilled`, `DmgDealt`, `DmgTaken`, `TurretDmgDealt`, `TotalGold`, `Lane`, `Win`, `item1`, `item2`, `item3`, `item4`, `item5`, `item6`, `kills`, `deaths`, `assists`, `PrimaryKeyStone`, `PrimarySlot1`, `PrimarySlot2`, `PrimarySlot3`, `SecondarySlot1`, `SecondarySlot2`, `SummonerSpell1`, `SummonerSpell2`, `CurrentMasteryPoints`, `EnemyChampionFk`, `DragonKills`, `BaronKills`,`visionScore`) VALUES (%s, %s , %s , %s , %s , %s , %s , %s , %s , %s , %s , %s , %s , %s , %s , %s , %s , %s , %s , %s , %s , %s , %s , %s , %s , %s , %s , %s , %s,%s)", (SummMatchId, cs, dmgDealt, dmgTaken, TurretDmgDealt, goldEarned, Role, win, Item1, Item2, Item3, Item4, Item5, Item6, kills, deaths, assists, PK1, PK2, PK3, PK4, SK1, SK2, spell1, spell2, masteryPoints, Enemy, dragonKills, baronKills,visionScore))
                    cursor.execute("INSERT INTO `TeamMatchTbl`(`MatchFk`, `B1Champ`, `B2Champ`, `B3Champ`, `B4Champ`, `B5Champ`, `R1Champ`, `R2Champ`, `R3Champ`, `R4Champ`, `R5Champ`, `BlueBaronKills`, `BlueRiftHeraldKills`, `BlueDragonKills`, `BlueTowerKills`, `BlueKills`, `RedBaronKills`, `RedRiftHeraldKills`, `RedDragonKills`, `RedTowerKills`, `RedKills`, `RedWin`, `BlueWin`)VALUES (%s , %s , %s, %s,%s , %s , %s, %s,%s , %s , %s, %s,%s , %s , %s, %s,%s , %s , %s,  %s, %s, %s, %s)",(MatchId,champList[0],champList[1],champList[2],champList[3],champList[4],champList[5],champList[6],champList[7],champList[8],champList[9],BaronKillsBlue,BlueRiftKills,DragonKillsBlue,towerKillsBlue,ChampionKillsBlue,BaronKillsRed,RedRiftKills,DragonKillsRed,towerKillsRed,ChampionKillsRed,RedWin,BlueWin,))
                    connection.commit()
                    print("NEXT NEXT NEXT")
                    i = i + 1
    except:
        pass

            
# Read summoner names and taglines from file
with open('summoner_names.txt', 'r', encoding='utf-8') as file:
    lines = file.readlines()
    for line in lines:
        summoner_name, tagline = line.strip().rsplit('#', 1)
        print(summoner_name)
        summoner(summoner_name, tagline)
        time.sleep(150)  # Adjust the sleep time as needed
