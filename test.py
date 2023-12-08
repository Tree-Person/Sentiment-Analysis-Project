from classes import *
from functions import *



def test_find_highest_element():
    lst1 = [1, 2, 3, 4, 5]
    expected1 = (5, 4)
    assert find_highest_element(lst1) == expected1

    lst2 = [-5, -4, -3, -2, -1]
    expected2 = (-1, 4)
    assert find_highest_element(lst2) == expected2

    lst3 = [-2, 0, 3, -5, 2]
    expected3 = (3, 2)
    assert find_highest_element(lst3) == expected3

    lst4 = [1, 2, 3, 3, 5]
    expected4 = (5, 4)
    assert find_highest_element(lst4) == expected4

    print("All test cases passed!")
   
def test_sentiment_analyzer():
    topic = "example"
    analyzer = SentimentAnalyzer(topic)

    # Test getDf method
    assert isinstance(analyzer.getDf(), pd.DataFrame)
    print("getDf method test passed!")

    # Test getMost method
    assert isinstance(analyzer.getMost(), str)
    print("getMost method test passed!")


test_find_highest_element()
test_sentiment_analyzer()