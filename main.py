import pandas
import webbrowser
from tempfile import NamedTemporaryFile

# number of rows from ratings.dat file which will be processed
NUMBER_OF_ROWS = 10000

# structure - UserID::MovieID::Rating::Timestamp
cols = ['UserID', 'MovieID', 'Rating', 'Timestamp']

data = pandas.read_csv('./data/ml-10M100K/ratings.dat', sep='::', header=None, nrows=NUMBER_OF_ROWS, engine='python', names=cols)
data = data.drop(cols[3], 1)
data = data.pivot(index=cols[0], columns=cols[1], values=cols[2])
data = data.fillna(0)

# debug
tmp = NamedTemporaryFile(delete=False, suffix='.html')
data.to_html(tmp)
webbrowser.open(tmp.name)


# TODO: Sort user-item matrix so that the density decreases along the main diagonal
