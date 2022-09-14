from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALS
from pyspark.sql import Row
from pyspark.sql.functions import collect_list

# Load up movie ID  -> moive name from dictionary

def load_movie_names():
    movie_names = {}
    
    with open("ml-100k/u.item") as f:
        for line in f:
            fields = line.split('|')
            movie_names[int(fields[0])] = fields[1].decode('ascii', 'ignore')
    return movie_names
