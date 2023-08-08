import pymongo 
from sklearn.feature_extraction.text import TfidfVectorizer
import gensim
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
import nltk
nltk.download('punkt')


MONGO_URL = 'mongodb+srv://yash23malode:9dtb8MGh5aCZ5KHN@cluster.u0gqrzk.mongodb.net/?retryWrites=true&w=majority'

myclient = pymongo.MongoClient(MONGO_URL)

mydb = myclient["prakat23"]

crawledsites = mydb["crawled_sites"]


text=""
for x in crawledsites.find():
    text=text+"url : "+x['url'] +'\n'+x['body']


tokenized_sentences = [word_tokenize(sentence) for sentence in text]

# Train Word2Vec model
model = Word2Vec(tokenized_sentences, vector_size=100, window=5, min_count=1, sg=0)

# Convert text data to embeddings using the trained Word2Vec model
embeddings = [model.wv[word] for sentence in tokenized_sentences for word in sentence]

# Now you have the embeddings for your text data
print(embeddings)
