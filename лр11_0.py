# Чтение json и формирование word2vec-модели
import pandas as pd, re
import multiprocessing, time
from gensim.models import Word2Vec
fn_wv = 'w2v.model'
def preprocess_s(s):
    s = s.lower()
    s = s.replace('ё', 'е')
    s = re.sub('[^а-я]', ' ', s)
    s = re.sub(' +', ' ', s)
    return s.strip()
corp = pd.read_json('corp.json', encoding = 'utf_8')
data = [] # Cписок предложений, предложение - список слов
for cont in corp.content:
    cont = re.sub('[;:!?]', '.', cont)
    sens = cont.split('.')
    for sen in sens:
        sen = preprocess_s(sen)
        if len(sen) > 0:
            words = sen.split()
            data.append(words) # len(data) = 9487
size, window, min_cnt, sg = 50, 2, 2, 0 # Используем модель CBOW
workers = multiprocessing.cpu_count()
n_iter = 150
t0 = time.time()
wv_model = Word2Vec(data, size = size, window = window, min_count = min_cnt,
                    sg = sg, workers = workers, iter = n_iter)
print('Время создания word2vec-модели:', round(time.time() - t0, 0))
print('word2vec-модель записана в файл', fn_wv)
wv_model.save(fn_wv)
