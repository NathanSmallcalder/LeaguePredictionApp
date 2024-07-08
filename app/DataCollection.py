
from RiotApiCalls import *
from championsRequest import *
from databaseQuries import *
import mysql.connector
import config   

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
    'user': 'root',
    'password': password,
    'host': 'mysql-db',
    'port': 3306,
    'database': 'LeagueStats',
    'buffered':'True'
}


connection = mysql.connector.connect(**config)

cursor = connection.cursor(buffered=True)
     
RegionStart = "europe"
Region = "EUW1"
summonerName = "El Maser"
tagline = "3616"
connection.autocommit = True
db_Info = connection.get_server_info()

Summoner = getPuuid(RegionStart,summonerName,tagline)
puuid = Summoner['puuid']
SummonerInfo = getSummonerDetails(Region,puuid)

SummId = SummonerInfo['id']

RankedDetails = getRankedStats(Region,SummId)
mastery = getMasteryStats(Region, puuid)
print(mastery)
#cursor.execute("INSERT INTO `SummonerUserTbl`(`SummonerName`) VALUES (%s )", (summonerName,))
connection.commit()

MatchIDs = requests.get("https://europe.api.riotgames.com/lol/match/v5/matches/by-puuid/"+ SummonerInfo['puuid'] +  "/ids?start=0&count=20&api_key=" + api_key)
MatchIDs = MatchIDs.json()
matchData = getMatches("euw1" ,Summoner,MatchIDs,SummonerInfo, RankedDetails, mastery)
matchData2 = getsMatchData()
Match = matchData2[1]['MatchIDS']
Match = Normalise(Match)

i = 0
for MatchId in MatchIDs:
    #MatchData
    MatchData = requests.get("https://europe.api.riotgames.com/lol/match/v5/matches/"+ MatchId +"?api_key=" + api_key)
    MatchData = MatchData.json()

    print("https://europe.api.riotgames.com/lol/match/v5/matches/"+ MatchId +"?api_key=" + api_key)

    ii = 0
    #Gets Champs (0-4 Team 1) , (5-9 Team 2)
    champList = []
    while ii < 10:
            champion = MatchData['info']['participants'][ii]['championName']
            cursor.execute("SELECT `ChampionId` FROM `ChampionTbl` WHERE `ChampionName` = (%s)", (champion, ))
            Champion = cursor.fetchall()
            Champion = Normalise(Champion)
            print("Champion", Champion, champion)
            champList.append(Champion)
            ii = ii + 1

    #100 or 0 blue team 
    BaronKillsBlue = int(MatchData['info']['teams'][0]['objectives']['baron']['kills'])
    ChampionKillsBlue = int(MatchData['info']['teams'][0]['objectives']['champion']['kills'])
    DragonKillsBlue = int(MatchData['info']['teams'][0]['objectives']['dragon']['kills'])
    BlueRiftKills = int(MatchData['info']['teams'][0]['objectives']['riftHerald']['kills'])
    ChampionKillsBlue = int(MatchData['info']['teams'][0]['objectives']['champion']['kills'])
    towerKillsBlue = int(MatchData['info']['teams'][0]['objectives']['tower']['kills'])
    BlueWin = int(MatchData['info']['teams'][0]['win'])
        
    #200 or 1 red team
    BaronKillsRed = int(MatchData['info']['teams'][1]['objectives']['baron']['kills'])
    ChampionKillsRed = int(MatchData['info']['teams'][1]['objectives']['champion']['kills'])
    DragonKillsRed  = int(MatchData['info']['teams'][1]['objectives']['dragon']['kills'])
    RedRiftKills = int(MatchData['info']['teams'][1]['objectives']['riftHerald']['kills'])
    ChampionKillsRed  = int(MatchData['info']['teams'][1]['objectives']['champion']['kills'])
    towerKillsRed  = int(MatchData['info']['teams'][1]['objectives']['tower']['kills'])
    RedWin = int(MatchData['info']['teams'][1]['win'])

    cursor.execute("SELECT `MatchFk` FROM `TeamMatchTbl` WHERE `MatchFk` = (%s)", (str(MatchId) ,))
    matchCheck = cursor.fetchone()
    print(matchData2)
    Match = matchData2[i]['MatchIDS']
    Patch = matchData2[i]['gameVersion']
    Rank = matchData2[i]['Rank']
    GameType = matchData2[i]['GameType']
    GameDuration = matchData[i]['GameDuration']

    Match = Normalise(Match)
    GameType = Normalise(GameType)
    Rank = Normalise(Rank)
    print("Rank", Rank)
    Patch = Normalise(Patch)

    cursor.execute("SELECT `RankId` FROM `RankTbl` WHERE `RankName` = (%s)", (Rank ,))
    RankId = cursor.fetchone()
    
    RankId = Rank[0]


    if matchCheck != None:
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

    print(Item1, " " , Item2,Item3, " " , Item4, " " ,Item5, " " , Item6)


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
    
    # Assuming the following variables are defined: cs, dmgDealt, dmgTaken, TurretDmgDealt, goldEarned, Role, win, Item1, Item2, Item3, Item4, Item5, Item6, kills, deaths, assists, PK1, PK2, PK3, PK4, SK1, SK2, spell1, spell2, masteryPoints, Enemy, dragonKills, baronKills
    cursor.execute("INSERT INTO `MatchStatsTbl`(`SummonerMatchFk`, `MinionsKilled`, `DmgDealt`, `DmgTaken`, `TurretDmgDealt`, `TotalGold`, `Lane`, `Win`, `item1`, `item2`, `item3`, `item4`, `item5`, `item6`, `kills`, `deaths`, `assists`, `PrimaryKeyStone`, `PrimarySlot1`, `PrimarySlot2`, `PrimarySlot3`, `SecondarySlot1`, `SecondarySlot2`, `SummonerSpell1`, `SummonerSpell2`, `CurrentMasteryPoints`, `EnemyChampionFk`, `DragonKills`, `BaronKills`) VALUES (%s, %s , %s , %s , %s , %s , %s , %s , %s , %s , %s , %s , %s , %s , %s , %s , %s , %s , %s , %s , %s , %s , %s , %s , %s , %s , %s , %s , %s)", (SummMatchId, cs, dmgDealt, dmgTaken, TurretDmgDealt, goldEarned, Role, win, Item1, Item2, Item3, Item4, Item5, Item6, kills, deaths, assists, PK1, PK2, PK3, PK4, SK1, SK2, spell1, spell2, masteryPoints, Enemy, dragonKills, baronKills))
    connection.commit()
    
    
    i = i + 1