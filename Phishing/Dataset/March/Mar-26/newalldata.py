# Import the necessary package to process data in JSON format
try:
    import json
except ImportError:
    import simplejson as json
 
import re
file=0
userslist=[]

quser={}
friend={}
i=0


tweets_filename='vtotal_benign_metadata_week1.txt'
# We use the file saved from last step as example

tweets_file = open(tweets_filename, "r")

for line in tweets_file:
    try:
        # Read in one line of the file, convert it into a json object 
        tweet = json.loads(line)

        pos=tweet["json_resp"]["resource"]
        
      
        with open('final_url_new1.txt', 'a') as f:
            f.write(str(pos))
            f.write("\n")

                
            
        
     
             
         
            

    except:
        # read in a line is not in JSON format (sometimes error occured)
        continue










