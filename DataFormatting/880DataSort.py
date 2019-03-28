
import pandas as pd
from os import listdir
import json

data = pd.read_csv('/Users/JeremyKulchyk/Downloads/stocknet-dataset-master/StockTable.txt',sep="\t")

Industries = set(list(data["Sector"]))
Symbols = sorted(list(data["Symbol"]))
Company = sorted(list(data["Company"]))


fileName = '/Users/JeremyKulchyk/Downloads/stocknet-dataset-master/tweet/raw/AAPL/2014-01-01.txt'

"""
Cols = ['created_at', 'text', 'user', 'retweet_count', 'favorite_count']
userCols = ['location','friends_count','followers_count','favourites_count']
"""

#TweetDict = {}
TweetDict = pd.DataFrame(columns=['Industry','Symbol','date','text','friends_count','followers_count','favourites_count','retweet_count','favorite_count','open','high','low','close','adjclose','volume'])
for sym in Symbols:
    Row = data.loc[data['Symbol'] == sym]
    Industry = Row.iloc[0]["Sector"]
    print(sym)
    Name = '/Users/JeremyKulchyk/Downloads/stocknet-dataset-master/tweet/raw/'+sym.replace("$","")
    try:
        files = listdir(Name)
    except:
        continue
    for file in files:
        fileName = Name + '/' + file
        date = file.replace(".txt","")
        priceName = '/Users/JeremyKulchyk/Downloads/stocknet-dataset-master/price/raw/'+sym.replace("$","")+'.csv'
        df = pd.read_csv(priceName)
        row = df.loc[df['Date'] == date]
        try:
            Open = row.iloc[0]["Open"]
            Close = row.iloc[0]["Close"]
            Low = row.iloc[0]["Low"]
            High = row.iloc[0]["High"]
            Volume = row.iloc[0]["Volume"]
            AdjClose = row.iloc[0]["Adj Close"]
        except:
            continue
        with open(fileName, "r") as line:
            for lines in line.readlines():
                tweets = json.loads(lines)
                text = tweets["text"]
                createdAt = tweets["created_at"]
                location = tweets["user"]["location"]
                friends_count = tweets["user"]["friends_count"]
                followers_count = tweets["user"]["followers_count"]
                favourites_count = tweets["user"]["favourites_count"]
                retweet_count = tweets["retweet_count"]
                favorite_count = tweets["favorite_count"]
                Len = len(TweetDict)
                TweetDict.loc[Len] = [Industry, sym.replace("$",""), date, text, friends_count, followers_count, favourites_count, retweet_count, favorite_count,Open, High, Low, Close, AdjClose, Volume]
        #print(file)
    
TweetDict.to_csv("/Users/JeremyKulchyk/Downloads/880DataSet.csv")
    
print(TweetDict.head())

