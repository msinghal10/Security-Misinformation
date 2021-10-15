# Import the necessary package to process data in JSON format
try:
    import json
except ImportError:
    import simplejson as json
import glob
file=0
screenname_list=[]

i=0


tweets_filename='combined-data.json' #add the json file here
# We use the file saved from last step as example

tweets_file = open(tweets_filename, "r")
# for file in glob.glob('All-conversation/*.json'):
#     tweets_file=open(file,"r")
for line in tweets_file:
    try:
        # Read in one line of the file, convert it into a json object 
        tweet = json.loads(line.strip())
        id_tweet=tweet['id']

        friends=tweet['user']['friends_count']
        status=tweet['user']['statuses_count']
        follow=tweet['user']['followers_count']
        h=tweet['entities']['urls']
        if len(h)>0:
            link = tweet['entities']['urls'][0]['url']
            if len(link) >0:
                link=1
            else:
                link=0
        else:
            link=0
        try:
            media=tweet['extended_entities']['media'][0]['type']
            if media == 'photo' or media =='video':
                media=1
            else:
                media=0
        except:
            media=0

        # picture=tweet['entities']['urls'][0]['url']
        screenname=tweet['user']['verified']
        if screenname == False:
            screenname = 0
        else:
            screenname= 1
        listed=tweet['user']['listed_count']
        descrip=tweet['user']['description']
        hel=len(descrip)
        image=tweet['user']['default_profile']
        if image == False:
            image=0
        else:
            image=1
        time_c=tweet['user']['created_at']
        ur=tweet['user']['url']
        if ur == None:
            ur=0
        else:
            ur=1
            # time_c=tweet['user']['created_at']
        if id_tweet not in screenname_list:
            with open('Descriptive_stats-newfea-all.csv','a') as f:
                f.write(str(id_tweet)+","+str(link)+","+str(media)+","+str(friends)+","+str(status)+","+str(follow)+","+str(screenname)+","+str(listed)+","+str(descrip)+","+str(image)+","+str(time_c)+","+str(ur))
                f.write("\n")
                screenname_list.append(id_tweet)
            
        
        #length=len(screenname_list) 
    except:
        # read in a line is not in JSON format (sometimes error occured)
        continue





 




