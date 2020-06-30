import sklearn
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.cluster import KMeans
import pymorphy2
from pymorphy2 import MorphAnalyzer
import numpy as np
import pandas as pd

"""грузим лист"""
keywords = [x.rstrip() for x in open('keywords.txt')]


"""грузим стопслова"""
stop_words = ['и', 'авито', 'от', 'эль', 'make', '2015', 'самый', 'украина', 'украину', 'иль', 'де', 'боте', '2016', '2017', 'в', 'с',
'й',
'ц',
'у',
'к',
'е',
'н',
'ш',
'щ',
'з',
'х',
'ъ',
'ф',
'ы',
'в',
'п',
'о',
'д',
'ж',
'э',
'я',
'ч',
'с',
'м',
'и',
'т',
'ь',
'б',
'ю',
'быть',
'не',
'да',
'по',
'который',
'на',
'весь',
'должный',
'это',
'для',
'она',
'он',
'от',
'без',
'только',
'тот',
'же',
'как',
'что',
'куда',
'почему',
'мочь',
'самый',
'раз',
'себя',
'так',
'год',
'или',
'наш',
'наше',
'чтоб',
'чтобы',
'2018',
'0','1','2','3','4','5','6','7','8','9','интернет', 'магазин', 'магазине', 'купить', 'из', 'до', 'женский', 'женская'
]


"""загружаем в переменную функцию стеммизации"""
m = pymorphy2.MorphAnalyzer()


"""класс анализа td idf + лемматизация"""
class LemmTfIdfVectorizer(TfidfVectorizer):
    def build_analyzer(self):
        analyzer = super(TfidfVectorizer, self).build_analyzer()
        return lambda x: (m.parse(word)[0].normal_form for word in analyzer(x))

"""заносим данные, корректировки (конфиг)"""
if __name__ == "__main__":
    tfidf_lemm_v = LemmTfIdfVectorizer(min_df=1, max_df=0.5, stop_words= stop_words)
    X = tfidf_lemm_v.fit_transform(keywords)

    k_means = KMeans(n_clusters=int(np.round(np.divide(len(keywords), 5))), 
        init='k-means++',
        n_init=10,
        max_iter=100,
        tol=0.0001,
        n_jobs=1)

    k_means.fit(X)

    dct = {}

    for key, label in zip(keywords, k_means.labels_):
        dct[label] = dct.get(label, [])+[key]

    with open('k-means.csv', 'w') as f:
        for cln in dct:
            for keyword in dct[cln]:
                f.write('{};{}\n'.format(cln, keyword))
