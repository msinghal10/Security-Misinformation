import sys
import urllib
import codecs
import json
import unicodecsv
import dateutil.parser as parser


# if len(sys.argv) != 3:  # the program name and the two arguments
#   # stop the program and print an error message
#   sys.exit("usage: tweet2csv.py <infile> <outfile>")

infile  = sys.argv[1]
outfile = sys.argv[2]

writer = unicodecsv.writer(open(outfile, mode='w'), encoding='utf-8', delimiter=',', quotechar='"', quoting=unicodecsv.QUOTE_NONNUMERIC)

for line in codecs.open(infile, 'r', 'utf8'):
  tweet = json.loads(line)
  # convert the created_at string to friendly ISO date
  # ts = (parser.parse(tweet['created_at']))

  row = []
  row.append(tweet['id'])
  row.append(tweet['full_text'])
  #Add more features as you want
  # row.append(tweet['source'])
  # row.append(ts.strftime("%Y-%m-%d %H:%M:%S"))
  # row.append(tweet['user']['id'])
  # row.append(tweet['user']['name'])
  # row.append(tweet['user']['description'])
  # row.append(tweet['user']['location'])
  # row.append(tweet['user']['followers_count'])
  # row.append(tweet['user']['friends_count'])
  # row.append(tweet['user']['profile_image_url'])

  writer.writerow(row)