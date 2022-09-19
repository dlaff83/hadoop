from dataclasses import fields
from json import load
from pyspark.sql import SparkSession
from pyspark.sql import Row
from pyspark.sql import functions

def load_movie_names():
    movie_names = {}
    with open("ml-100k/u.item") as f:
        for line in f:
            fields = line.split('|')
            movie_names[int(fields[0])] = fields[1]
        return movie_names

def parse_input(line):
    fields = line.split()
    #returning a row object instead of a tuple
    return Row(movie_id = int(fields[1]), rating = float(fields[2]))

if __name__ == "__main__":
    #Create spark session
    spark = SparkSession.builder.appName("PopularMovies").getOrCreate()

    #Load up our movie ID -> name dictionary
    movie_names = load_movie_names()

    #Get raw data
    lines = spark.sparkContext.textFile("hdfs:///user/maria_dev/ml-100k/u.data")

    #Convert raw data to RDD of Row objects with (movie_id, rating)
    movies = lines.map(parse_input)

    #Convert RDD into dataframe 
    movies_dataset = spark.createDataFrame(movies)

    #Compute average rating for each movie_id
    average_ratings = movies_dataset.groupBy('movie_id').avg('rating')

    #Compute count of ratings for each movie_id
    counts = movies_dataset.groupBy('movie_id').count()

    #Join the two two aggregated together 
    average_and_count = counts.join(average_ratings, 'movie_id')

    #top ten results
    top_ten = average_and_count.orderBy('avg(rating)').take(10)

    for movie in top_ten:
        print(movie_names[movie[0]], movie[1], movie[2])
    
    spark.stop()
