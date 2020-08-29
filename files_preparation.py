from functools import partial
from multiprocessing import Pool
from operator import itemgetter
import pickle
import string

from bs4 import BeautifulSoup
from gensim.models.fasttext import FastText
from imdb import IMDb
from nltk.stem import WordNetLemmatizer
import numpy as np
import requests


lemmatizer = WordNetLemmatizer()
exclude = set(string.punctuation)
model = FastText.load('storage/imdb_model')


def return_id_from_title(title):
    r = requests.get(f"https://www.imdb.com/find?q={title.replace(' ', '+')}")
    bs = BeautifulSoup(r.text, 'html.parser')
    for result in bs.find_all('tr'):
        if '/tt' in result.a['href']:
            href = result.a['href']
            break

    return href[href.find('/tt')+3:-1]


def return_title_from_id(id):
    r = requests.get('https://www.imdb.com/title/tt' + str(id))
    bs = BeautifulSoup(r.text, 'html.parser')
    title = bs.find('title').text 
    return title[:title.find(' - IMDb')]



def get_description_by_id(id):
    movie = IMDb().get_movie(id)
    movie_description = movie.get('plot')
    movie_description = "".join(i for i in movie_description if i not in exclude)
    return movie_description.lower().replace('(', ' ').replace(')', ' ').replace('.', ' ')


def get_smart_word_vec(word):
    word_list = word.split(' ')
    for i in range(len(word_list)):
        word_list[i] = lemmatizer.lemmatize(word_list[i])
    new_str = ' '.join(word_list)
    return model.wv.word_vec(new_str)


def cosine_similarity(v1, v2):
    return (v1 @ v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))


def find_neighbours(id, id_list, description_vectors, calculated_vectors):
    neighbours = []
    vector = calculated_vectors[id_list.index(id)] if id in id_list else get_smart_word_vec(get_description_by_id(id))
    with Pool(2) as p:
        help_neighbours = p.map(partial(cosine_similarity, v2=vector), description_vectors)
    for i in range(len(help_neighbours)):
        neighbours.append((i, help_neighbours[i]))
    neighbours.sort(key=itemgetter(1))
    neighbours.reverse()
    for i in range(20):
        neighbours[i] = neighbours[i][0]
    return neighbours[:20]


if __name__ == "__main__":

    string = ''
    with open('storage/new_better_descriptions.txt', 'r') as f:
        string += f.read()

    descr = string.split('\n')
    descr = [x for x in descr if x[:x.find('%$')+3:x.find('@@@')] != '']
    link_list, name_list = [], []

    for i in descr:
        name_list.append(i[:i.find('%$')])
        link_list.append(i[i.find('@@@')+3:])

    for i in range(len(descr)):
        descr[i] = descr[i][descr[i].find('%$')+3:descr[i].find('@@@')]

    id_list = [i[i.find('/tt')+3:-1] for i in link_list][:1530]

    with open('id_list.pkl', 'wb') as file:
        pickle.dump(id_list, file)

    with open('name_list.pkl', 'wb') as file:
        pickle.dump(name_list, file)

    with open('link_list.pkl', 'wb') as file:
        pickle.dump(link_list, file)

