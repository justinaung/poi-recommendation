import os
from time import time
import pickle
import argparse

import pandas as pd
from py2neo import Graph


USER = os.environ.get('NEO4J_USER', 'neo4j')
PASS = os.environ.get('NEO4J_PASSWORD', 'neo4j')
BOLT_URL = os.environ.get('NEO4J_BOLT_URL', 'bolt://localhost:7687')


parser = argparse.ArgumentParser()
saved_records_file = parser.add_argument('saved_records_file', metavar='path', type=str)
test_file = parser.add_argument('test_file', metavar='path', type=str)
neighbourhood_size = parser.add_argument('--neighbourhood_size', type=int, default=100)
num_records = parser.add_argument('--num_records', type=int, default=10)
args = parser.parse_args()


def main(saved_records_file, test_file, neighbourhood_size=100, num_records=10):
    graph = Graph(BOLT_URL, auth=(USER, PASS), secure=True)
    returned_records = user_user_collaborative_filtering(graph, neighbourhood_size, num_records)

    with open(saved_records_file, 'wb') as handle:
       pickle.dump(returned_records, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print(f'Evaluation with {test_file} for neighbourhood ({neighbourhood_size}) and records ({num_records})')
    evaluate(returned_records, test_file)


def user_user_collaborative_filtering(graph, neighbourhood_size, num_records):
    query = """
        // Get user pairs and count of distinct places that they have checked-in
        MATCH (u1:User)-[c1:CHECKED_IN]->(p:Place)<-[c2:CHECKED_IN]-(u2:User)
        WHERE u1 <> u2
        WITH u1, u2, COUNT(DISTINCT p) as intersection_count

        // Get count of all the distinct places that they have checked-in between them
        MATCH (u:User)-[:CHECKED_IN]->(p:Place)
        WHERE u in [u1, u2]
        WITH u1, u2, intersection_count, COUNT(DISTINCT p) as union_count

        // Compute Jaccard index
        WITH u1, u2, intersection_count, union_count, (intersection_count * 1.0/union_count) as jaccard_index

        // Get top k neighbours based on Jaccard index
        ORDER BY jaccard_index DESC, u2.userId
        WITH u1, COLLECT(u2)[0..$k] as neighbours
        WHERE SIZE(neighbours) = $k

        UNWIND neighbours as neighbour
          WITH u1, neighbour

          // Get top n recommendations from the selected neighbours
          MATCH (neighbour)-[:CHECKED_IN]->(p:Place)
          WHERE not (u1)-[:CHECKED_IN]->(p)
          WITH u1, p, COUNT(DISTINCT neighbour) as cnt
          ORDER BY u1.userId, cnt DESC
          RETURN u1.userId as user, COLLECT(p.placeId)[0..$n] as records
    """
    records = {}
    start = time()

    print('Running the Cypher query...')
    for row in graph.run(query, k=neighbourhood_size, n=num_records):
        records[row[0]] = row[1]

    end = time()
    print('Time taken: ', end - start)
    return records


def evaluate(returned_records, test_file):
    test_df = pd.read_csv(test_file, names=['user_id', 'place_id'], sep='\t', header=0)
    test_df['user_id'] = test_df['user_id'].astype(int)
    test_df['place_id'] = test_df['place_id'].astype(int)

    hit_count = 0
    pred_count = 0
    actual_count = 0

    for k, v in returned_records.items():
        returned_places = set([int(x) for x in v])
        actual_places = set(test_df[test_df['user_id'] == int(k)]['place_id'])
        hits = returned_places.intersection(actual_places)
        hit_count += len(hits)
        pred_count += len(returned_places)
        actual_count += len(actual_places)

    precision = hit_count / pred_count
    recall = hit_count / actual_count
    f1 = 2 * ((precision * recall) / (precision + recall))

    print('Precision: ', precision)
    print('Recall: ', recall)
    print('F1: ', f1)


if __name__ == '__main__':
    main(args.saved_records_file, args.test_file, args.neighbourhood_size, args.num_records)
