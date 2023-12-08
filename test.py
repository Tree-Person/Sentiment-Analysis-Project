from classes import *
from functions import *


water = SentimentAnalyzer("water")
dfs = water.getDf()

info = []
for i in dfs['Score']:
        if i >= 1:
            info.append('positive')
        elif 0 < i < 1:
            info.append('neutral')
        else:
            info.append('negative')

dfs['Results'] = info

def find_highest_element(lst):
    max_ele = max(lst)
    for i , x in enumerate(lst):
        if x == max_ele:
            return max_ele , i
        
lst = [1, 2, 3, 4, 5]
print(find_highest_element(lst))