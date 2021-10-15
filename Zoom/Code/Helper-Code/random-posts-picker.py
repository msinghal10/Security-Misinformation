import pandas as pd
import numpy as np
import random

# For any files that are in csv format use lines 6 to 11
df = pd.read_csv("Comined-without-new.csv")
# print('Source DataFrame:\n', df)

df2=df.sample(n=100,random_state = None)
print(df2)
df2.to_csv('Random-sample-100-instagram.csv', index = False)

# For any files that is in json format use the lines from 12 to 28
# selected=[]
# i=0

# limit=101 #No of tweets to select 
# lines = open('combined-data.json').read().splitlines()       #the read()splitlines is for json, so we can remove that.
# while(i<limit):
# 	print("Choosen so far:"+str(i)+"/"+str(limit))
# 	selected_tweet =random.choice(lines)
# 	if selected_tweet not in selected:
# 		selected.append(selected_tweet)
# 		# print("Selected:"+str(selected_tweet))
# 		file=open("combinedrandomdatatwitter-100.json",'a')
# 		file.write(selected_tweet+"\n")
		
# 		i=i+1
# 	else:
# 		print("Ignored")