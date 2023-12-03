from functions import sentiment as sentiment_function

class SentimentAnalyzer:
    def __init__(self, topic):
        self.topic = topic
        self.data = sentiment_function(topic) 

    def getDf(self):
        return self.data

# Example usage
water = SentimentAnalyzer('water')
print(water)