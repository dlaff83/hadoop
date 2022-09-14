from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALS
from pyspark.sql import Row
from pyspark.sql.functions import lit

# Load up movie ID  -> moive name from dictionary
def load_movie_names():
    movie_names = {}
    
    with open("ml-100k/u.item") as f:
        for line in f:
            fields = line.split('|')
            movie_names[int(fields[0])] = fields[1].decode('ascii', 'ignore')
    return movie_names

# Convert u.data lines into (user_id,move_id,rating) rows

def parse_input(line):
    fields = line.value.split()
    return Row(user_id = int(fields[0]), movie_id = int(fields[1]), rating = float(fields[2]))

if __name__ == '__main__':
    # Create a SparkSession
    spark = SparkSession.builder.appName("movie_recs").getOrCreate()

    # Load up our movie ID -> name directory
    movie_names = load_movie_names()

    # Get the raw data
    lines = spark.read.text("hdfs:///user/maria_dev/ml-100k/u.data").rdd

    # Convert it to an RDD of row objects with (user_id,movie_id,rating)
    ratings_rdd = lines.map(parse_input)

    # Convert to a DF and cache it
    ratings = spark.createDataFrame(ratings_rdd).cache()
    
    # Create an ALS collaborative filtering model from the complete data set
    als = ALS(maxIter=5, regParam=0.01, userCol="user_id", itemCol="move_id", ratingCol="rating")
    model = als.fit(ratings)

    # Print out ratings from user 0:
    print("\nRatings for user ID 0:")
    user_ratings = ratings.filter("user_id = 0")
    for rating in user_ratings.collect():
        print(movie_names[rating['movie_id']], rating['rating'])
    
    print("\nTop 20 recommendations:")
    # Find movies rated more than 100 times
    rating_counts = ratings.groupBy('movie_id').count().filter('count > 100')

    # Construct a "test" DF for user 0 with every movie rated more than 100 times
    popular_movies = rating_counts.select('moive_id').withColumn('user_id', lit(0))

    # Run model on list of popular movies for user id of 0
    recommendations = model.transform(popular_movies)

    # Get the top 20 movies with the highest predicted rating for this user
    top_recommendations = recommendations.sort(recommendations.prediction.desc()).take(20)

    for recommendation in top_recommendations:
        print(movie_names[recommendation['movie_id']], recommendation['prediction'])

    spark.stop()
