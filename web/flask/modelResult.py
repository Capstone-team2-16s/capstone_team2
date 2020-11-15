import model
    
request_id = '@AdinaPorter'
# 문장 전처리
model.preproc_Sentence()
tweets = model.preproc_Sentence.readTweets(request_id)
tweets_data = model.preproc_Sentence.preprocTweets(tweets)
# 트위터 감정 분석
df_res = model.tweet_SentimentAnalyse.sentimentAnalyse(tweets_data) #전체 분석 결과
typeRatio = model.tweet_SentimentAnalyse.countTypes(df_res) #문장별 비율
# 단어 카운트
countWord = model.word_COUNT.countWord(request_id)

#merge dict
#typeRatio + countWord + 문장예시도 추가해야
resDict = {}
resDict.update(typeRatio)
resDict.update(countWord)
# print(resDict)
