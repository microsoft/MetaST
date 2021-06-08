"""
This is a simple application for sentence embeddings: semantic search

We have a corpus with various sentences. Then, for a given query sentence,
we want to find the most similar sentence in this corpus.

This script outputs for various queries the top 5 most similar sentences in the corpus.
"""
from sentence_transformers import SentenceTransformer
import scipy.spatial
import io
import argparse

def parser_config(parser):

    
    parser.add_argument("--query_input",
                        default='sample_data/targetset_raw.txt',
                        type=str,
                        required=False,
                        help="Path to bert config")

    parser.add_argument("--corpus_input",
                        default='lm_data/train.txt',
                        type=str,
                        required=False,
                        help="Path to bert config")



    return parser.parse_args()


parser = argparse.ArgumentParser()
args = parser_config(parser)


embedder = SentenceTransformer('bert-base-nli-mean-tokens')

# Corpus with example sentences
'''
corpus = ['A man is eating a food.',
          'A man is eating a piece of bread.',
          'The girl is carrying a baby.',
          'A man is riding a horse.',
          'A woman is playing violin.',
          'Two men pushed carts through the woods.',
          'A man is riding a white horse on an enclosed ground.',
          'A monkey is playing drums.',
          'A cheetah is running behind its prey.'
          ]
'''

corpus = []
with open(args.corpus_input) as f:
    for line in f:
        #print (line.strip())
        corpus.append(line.strip())

corpus_embeddings = embedder.encode(corpus)

# Query sentences:
#queries = ['Send message to Idan Haim.', 'Someone in a gorilla costume is playing a set of drums.', 'A cheetah chases prey on across a field.']

queries = []
with open(args.query_input) as f:
    for line in f:
        #print (line.strip())
        queries.append(line.strip())

query_embeddings = embedder.encode(queries)

# Find the closest 5 sentences of the corpus for each query sentence based on cosine similarity
closest_n = 5

f_output =  open("outputs/outputs.txt",'w') 

for query, query_embedding in zip(queries, query_embeddings):
    distances = scipy.spatial.distance.cdist([query_embedding], corpus_embeddings, "cosine")[0]

    results = zip(range(len(distances)), distances)
    results = sorted(results, key=lambda x: x[1])

    print("\n\n======================\n\n")
    print("Query:", query)
    print("\nTop 5 most similar sentences in corpus:")

    for idx, distance in results[0:closest_n]:
        print(corpus[idx].strip(), "(Score: %.4f)" % (1-distance))
        f_output.write(corpus[idx].strip() + '\n') 



