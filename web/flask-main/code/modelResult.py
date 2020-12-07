import model
import pandas as pd
import io

def strToDF(data):
    file = io.StringIO(data) #string to dataframe으로 쓸 수 있게 임시 파일 형식으로
    df_tweets = pd.read_csv(file,sep="\n",names=['data'])[0:700] #preproc_sentence에 쓸 dataframe 700문장까지만 로드
    file.close()
    return df_tweets

def analysisResult(tweets): #string 형식으로 받아옴
    # 문장 전처리
    model.preproc_Sentence()

    #string to dataframe
    df_tweets = strToDF(tweets)

    # 감성 분석
    tweets_data = model.preproc_Sentence.preprocTweets(df_tweets)
    df_res = model.tweet_SentimentAnalyse.sentimentAnalyse(tweets_data) #전체 분석 결과
    typeRatio = model.tweet_SentimentAnalyse.countTypes(df_res) #문장별 비율

    # 단어 카운트
    countWord = model.word_COUNT.countWord(tweets)

    # merge dict
    resDict = {}
    resDict.update(typeRatio)
    resDict.update(countWord)
    return resDict
