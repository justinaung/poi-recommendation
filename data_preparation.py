import pandas as pd


train_df = pd.read_csv('data/Foursquare_train.txt', sep='\t')

train_users = train_df.loc[:, ['userID']].rename({'userID': 'userId:ID(User-ID)'}, axis=1)
train_users = train_users.drop_duplicates()
train_users[':LABEL'] = 'User'

train_places = train_df.loc[:, ['placeID']].rename({'placeID': 'placeId:ID(Place-ID)'}, axis=1)
train_places = train_places.drop_duplicates('placeId:ID(Place-ID)')
train_places[':LABEL'] = 'Place'

train_checkins = (
    train_df.loc[:, ['userID', 'placeID']]
    .rename({
        'userID': ':START_ID(User-ID)',
        'placeID': ':END_ID(Place-ID)',
    }, axis=1)
)
train_checkins[':TYPE'] = 'CHECKED_IN'

train_users.to_csv('data/neo4j/train_users.csv', index=False)
train_places.to_csv('data/neo4j/train_places.csv', index=False)
train_checkins.to_csv('data/neo4j/train_checkins.csv', index=False)
