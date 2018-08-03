from tweepy import Stream
from tweepy import OAuthHandler
from tweepy.streaming import StreamListener
import json
import sentiment_mod as s

#consumer key, consumer secret, access token, access secret.
ckey="mZIiraQNLRm6CxI0mgE5hjWkO"
csecret="ECZK58Rop1WaQXSbxQD32aKqmtX2ig72i6xBklU4DGsEFx89DB"
atoken="944869562993983489-1KxQ2rWdvRGbigdSoTVvPVnUPZ2b9sr"
asecret="HgmAZQ4fpcDdFZ4bbjrK0briLLHK3RQDiWQsoNxynqlZI"

class listener(StreamListener):
    def on_data(self,data):
        all_data=json.loads(data)
        tweet=all_data["text"]
        sentiment_value,confidence=s.sentiment(tweet)
        print(tweet,sentiment_value,confidence)

        if confidence*100>=80:
            output=open('twitter-out.txt','a') #a:Stand for append
            output.write(sentiment_value)
            output.write('\n')
            output.close()
        return True
    def on_error(self,status):
        print(status)

auth = OAuthHandler(ckey, csecret)
auth.set_access_token(atoken, asecret)

twitterStream = Stream(auth, listener())
twitterStream.filter(track=["meri pyaari bindu"])
