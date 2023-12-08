from functions import final_output as sentiment_function

class SentimentAnalyzer:
    def __init__(self, topic):
        self.topic = topic  # Initializes the 'topic' attribute with the value passed as an argument
        self.data = sentiment_function(topic)  # Calls the 'sentiment_function' with the 'topic' argument and assigns the returned value to the 'data' attribute

    def getDf(self):
        return self.data  # Returns the 'data' attribute

    def getMost(self):
        return self.data['Results'].value_counts().idxmax()  # Returns the most frequent value in the 'Results' column of the 'data' attribute
