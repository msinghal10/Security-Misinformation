import re
import glob
try:
    import json
except ImportError:
    import simplejson as json

file=0
screenname_list=[]

i=0
j=1

# tweets_filename='2020-alltweets.json'
# # We use the file saved from last step as example

# tweets_file = open(tweets_filename, "r")
# for line in tweets_file:
# 	try:
#         # Read in one line of the file, convert it into a json object 
# 		tweet = json.loads(line.strip())
# 		tweet_id=tweet['lang']
# 		print(tweet_id)
# 		if tweet_id == 'en':
# 			with open('filtered-2020.json', 'a') as f:
# 				f.write(line)
# 			# f.write("\n")
# 			# i=1
# 				print("Lang was EN")
# 		else:
# 			continue
# 		# 	i=0
# 		# 	print("lang was Else")
# 		# with open('lang_All_conversations_Tweet_Id_list.csv', 'a') as f:
# 		# 	f.write(str(tweet1)+","+str(i))
# 		# 	fe.write("\n")
# 					# screenname_list.append(id_user)
#         #length=len(screenname_list) 
# 	except:
# 		continue

tweets_filename = 'selected_tweets.json'
tweets_file = open(tweets_filename, 'r')

words = '|'.join([ 'hxxp', 'hxxps', 'hXXp', 'hXXps'])

#print(words_re)

words_re = re.compile(words)

for line in tweets_file:
    try:
        tweet = json.loads(line.strip())
        tweet_text = tweet['full_text']
        if words_re.search(tweet_text):
        	i=i+1
            # with open('keyword-filtered-2020.json', 'a') as f:
            #     f.write(line)
		
    except:
        continue

print(i)
