import model

def analysisResult(tweets):
    request_id = '@AdinaPorter'
    # 문장 전처리
    model.preproc_Sentence()
    #tweets = model.preproc_Sentence.readTweets(request_id) => 기존 메서드 제외
    tweets_data = model.preproc_Sentence.preprocTweets(tweets)
    # 트위터 감정 분석
    df_res = model.tweet_SentimentAnalyse.sentimentAnalyse(tweets_data) #전체 분석 결과
    typeRatio = model.tweet_SentimentAnalyse.countTypes(df_res) #문장별 비율
    # 단어 카운트
    countWord = model.word_COUNT.countWord(tweets)

    #merge dict
    resDict = {}
    resDict.update(typeRatio)
    resDict.update(countWord)
    return resDict
