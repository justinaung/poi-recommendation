# POI Recommendation with User-User Collaborative Filtering

## Creating data
Before running the commands below, create a Neo4j graph database. Export the Neo4j database home path.
```
chmod +x create_graph.sh
./create_graph
```

## Run recommendation
```
python recommendation.py returned_records.pickle data/Foursquare_test.txt --neighbourhood_size=100 --num_records=10
```
