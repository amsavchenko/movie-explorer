import pickle

from flask import Flask, render_template, request, redirect, url_for
import numpy as np

from files_preparation import return_id_from_title, return_title_from_id, find_neighbours


with open('storage/id_list.pkl', 'rb') as file:
    id_list = pickle.load(file)

with open('storage/name_list.pkl', 'rb') as file:
    name_list = pickle.load(file)

with open('storage/link_list.pkl', 'rb') as file:
    link_list = pickle.load(file)

description_vectors = np.load('storage/vectors.npy')
calculated_vectors = np.load('storage/plot_vectors.npy')


app = Flask(__name__)
app.secret_key = b'_5#y2L"F4Q8z\n\xec]/'


@app.route('/')
def start():
    return render_template('start_page.html')


@app.route('/recommendation/', methods=['POST', 'GET'])
def help():
    if request.method == 'POST':
        title = request.form['Title']
        return redirect(url_for('get_recommendation', id=return_id_from_title(title)))


@app.route('/get_recommendation/<id>', methods=['GET'])
def get_recommendation(id):
    title = return_title_from_id(id)
    neighbours = find_neighbours(id, id_list, description_vectors, calculated_vectors)
    return render_template('recommendation_page.html', title=title, id_list1=neighbours[:len(neighbours)//2],
                           id_list2=neighbours[len(neighbours)//2:], name_list=name_list, link_list=link_list)


if __name__ == "__main__":
    app.run()