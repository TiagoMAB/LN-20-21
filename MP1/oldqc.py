import nltk
from sklearn import svm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multiclass import OneVsRestClassifier

def read_labels_coarse(doc):

    file = open(doc, "r")
    labels = []
    for sentence in file:
        label = sentence.split(":")[0]
        if label not in labels:
            labels += [label]

    return labels

def read_labels_fine(doc):

    file = open(doc, "r")
    labels = []
    for sentence in file:
        label = sentence.split(" ")[0]
        if label not in labels:
            labels += [label]

    return labels

#Returns all labels (not unique) in a doc
def read_all_labels(doc):

    file = open(doc, "r")
    labels = []
    for sentence in file:
        label = sentence.split(":")[0]
        labels += [label]

    return labels

#Returns all labels fine (not unique) in a doc
def read_all_labels_fine(doc):

    file = open(doc, "r")
    labels = []
    for sentence in file:
        label = sentence.split(" ")[0]
        labels += [label.split("\n")[0]]

    return labels

def get_bigrams(question):

    tokenized_question = nltk.word_tokenize(question)
    bigram_list = []

    for i in range(len(tokenized_question) - 1):
        bigram_list += [(tokenized_question[i], tokenized_question[i+1])]
    for i in range(len(tokenized_question)):
        bigram_list += [(tokenized_question[i], )]

    return bigram_list

#returns questions in a file that doesn't have labels before questions
def read_questions_only(doc):

    file = open(doc, "r")
    token_list = []
    for question in file:
        token_list += [nltk.word_tokenize(question)]

    return token_list

#returns questions in a file has labels before questions
def read_questions(doc):

    file = open(doc, "r")
    token_list = []
    for sentence in file:
        question = sentence.split(" ", 1)[1]
        token_list += [nltk.word_tokenize(question)]

    print(token_list)
    return token_list


def most_frequent_words(list):

    dic = {}
    for vector in list:
        for word in vector:
            if word in dic:
                dic[word] += 1
            else:
                dic[word] = 1

    return {k: v for k, v in sorted(dic.items(), key=lambda item: item[1])}

def get_words(list):

    dic = {}
    for word in list:
        if word in dic:
            dic[word] += 1
        else:
            dic[word] = 1

    return {k: v for k, v in sorted(dic.items(), key=lambda item: item[1])}

def get_train_set(labels, questions):

    train_set = []
    for i in range(len(questions)):
        train_set += [(get_words(questions[i]), labels[i])]

    return train_set

#returns test set for classifier
def get_test_set(doc):
    questions = pre_processing(read_questions_only(doc))

    test_set = []
    for i in range(len(questions)):
        test_set += [get_words(questions[i])]

    return test_set


def evaluate(expected, result):

    accuracy = 0
    ABBR_failure = 0
    other_failure = 0
    for i in range(len(expected)):
        if (expected[i] == result[i][0]):
            accuracy += 1
        else:
            if result[i] == "ENTY:other":
                ABBR_failure += 1
            else:
                other_failure += 1
            #print("Failed in this case:" + expected[i] + " " + result[i][0] + " " + str(i) + "\n")

    accuracy /= len(expected)
    print(accuracy)
    print(ABBR_failure)
    print(other_failure)
    return

def pre_processing(token_list):

    stop_words = [",", "'", "\"", "`", "´", "?", "What", "what", "for"]

    processed_list = []
    for vector in token_list:
        processed_token = []
        for word in vector:
            if word not in stop_words:
                processed_token += [word]
        processed_list += [processed_token]
    print(processed_list)
    return processed_list


def classify_HUM():

    return

def classify_NUM():

    return

def classify_LOC():

    return

def classify_DESC():

    return

def classify_ABBR():

    return

def classify_ENTY():

    return

from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer

class StemmaTokenizer:
    def __init__(self):
        self.wnl = WordNetLemmatizer()
        self.ps = PorterStemmer()
    def __call__(self, doc):
        return [self.ps.stem(o) for o in [self.wnl.lemmatize(t) for t in word_tokenize(doc)]]

def SVM(train_doc, labels_doc, test_doc, test_labels, fine):

    if fine:
        corpus = open(train_doc, "r")
        vectorizer = TfidfVectorizer(strip_accents = 'unicode', stop_words = ["what", "a", "on", "``", "´´", "and", "to", ":", "`", "´"], min_df = 3, tokenizer=StemmaTokenizer())
        X = vectorizer.fit_transform(corpus)

        clf = svm.SVC()
        clf.fit(X, read_all_labels(labels_doc))

        test = open(test_doc, "r")
        test_set = vectorizer.transform(test)
        coarse_classifications = clf.predict(test_set)

        labels = read_all_labels(labels_doc)
        labels_fine = read_all_labels_fine(labels_doc)
        doc_fine = open(train_doc, "r")
        corpus_fine = []
        for sentence in doc_fine:
            corpus_fine += [sentence.split("\n")[0]]

        QUESTIONS_LOC = []
        QUESTIONS_ENTY = []
        QUESTIONS_DESC = []
        QUESTIONS_HUM = []
        QUESTIONS_NUM = []
        QUESTIONS_ABBR = []
        Y_LOC = []
        Y_ENTY = []
        Y_DESC = []
        Y_HUM = []
        Y_NUM = []
        Y_ABBR = []

        for i in range(len(labels)):
            if labels[i] == "LOC":
                QUESTIONS_LOC += [corpus_fine[i]]
                Y_LOC += [labels_fine[i]]
            elif labels[i] == "ENTY":
                QUESTIONS_ENTY += [corpus_fine[i]]
                Y_ENTY += [labels_fine[i]]
            elif labels[i] == "DESC":
                QUESTIONS_DESC += [corpus_fine[i]]
                Y_DESC += [labels_fine[i]]
            elif labels[i] == "HUM":
                QUESTIONS_HUM += [corpus_fine[i]]
                Y_HUM += [labels_fine[i]]
            elif labels[i] == "NUM":
                QUESTIONS_NUM += [corpus_fine[i]]
                Y_NUM += [labels_fine[i]]
            else:
                QUESTIONS_ABBR += [corpus_fine[i]]
                Y_ABBR += [labels_fine[i]]

        print(QUESTIONS_ENTY)
        vectorizer_X_LOC = TfidfVectorizer(strip_accents = 'unicode', min_df = 3, tokenizer=StemmaTokenizer())
        X_LOC = vectorizer_X_LOC.fit_transform(QUESTIONS_LOC)
        vectorizer_X_ENTY = TfidfVectorizer(strip_accents = 'unicode', stop_words = ["as", "by", "the", "a", "on", "``", "´´", "and", "to", ":", "`", "´"], min_df = 3, tokenizer=StemmaTokenizer()) #stop words better here
        X_ENTY = vectorizer_X_ENTY.fit_transform(QUESTIONS_ENTY)
        vectorizer_X_DESC = TfidfVectorizer(strip_accents = 'unicode', min_df = 3, tokenizer=StemmaTokenizer())
        X_DESC = vectorizer_X_DESC.fit_transform(QUESTIONS_DESC)
        vectorizer_X_HUM = TfidfVectorizer(strip_accents = 'unicode', min_df = 3, tokenizer=StemmaTokenizer())
        X_HUM = vectorizer_X_HUM.fit_transform(QUESTIONS_HUM)
        vectorizer_X_NUM = TfidfVectorizer(strip_accents = 'unicode', min_df = 3, tokenizer=StemmaTokenizer()) #current list of spotwords damages results here
        X_NUM = vectorizer_X_NUM.fit_transform(QUESTIONS_NUM)
        vectorizer_X_ABBR = TfidfVectorizer(strip_accents = 'unicode', min_df = 3, tokenizer=StemmaTokenizer())
        X_ABBR = vectorizer_X_ABBR.fit_transform(QUESTIONS_ABBR)

        clf_LOC = OneVsRestClassifier(svm.SVC())
        clf_LOC.fit(X_LOC, Y_LOC)
        clf_ENTY = OneVsRestClassifier(svm.SVC())
        clf_ENTY.fit(X_ENTY.toarray(), Y_ENTY)
        clf_DESC = OneVsRestClassifier(svm.SVC())
        clf_DESC.fit(X_DESC, Y_DESC)
        clf_HUM = OneVsRestClassifier(svm.SVC())
        clf_HUM.fit(X_HUM, Y_HUM)
        clf_NUM = svm.SVC()
        clf_NUM.fit(X_NUM, Y_NUM)
        clf_ABBR = OneVsRestClassifier(svm.SVC())
        clf_ABBR.fit(X_ABBR, Y_ABBR)

        predictions = []

        test = open(test_doc, "r")
        test_questions = []
        for question in test:
            test_questions += [question]


        for i in range(len(coarse_classifications)):
            if coarse_classifications[i] == "LOC":
                predictions += [clf_LOC.predict(vectorizer_X_LOC.transform([test_questions[i]]))]
            elif coarse_classifications[i] == "ENTY":
                predictions += [clf_ENTY.predict(vectorizer_X_ENTY.transform([test_questions[i]]).toarray())]
            elif coarse_classifications[i] == "DESC":
                predictions += [clf_DESC.predict(vectorizer_X_DESC.transform([test_questions[i]]))]
            elif coarse_classifications[i] == "HUM":
                predictions += [clf_HUM.predict(vectorizer_X_HUM.transform([test_questions[i]]))]
            elif coarse_classifications[i] == "NUM":
                predictions += [clf_NUM.predict(vectorizer_X_NUM.transform([test_questions[i]]))]
            else:
                predictions += [clf_ABBR.predict(vectorizer_X_ABBR.transform([test_questions[i]]))]

        evaluate(read_all_labels_fine(test_labels), predictions)
        #print(clf.score(predictions, read_all_labels_fine(test_labels)))
    else:

        corpus = open(train_doc, "r")
        vectorizer = TfidfVectorizer(strip_accents = 'unicode', stop_words = ["other", "thi","but", "'s", "if", "by", "as", "it", "these", "what", "a", "on", "``", "´´", "and", "to", ":", "`", "´"], min_df = 3, tokenizer=StemmaTokenizer())
        X = vectorizer.fit_transform(corpus)
        clf = svm.SVC()
        clf.fit(X, read_all_labels(labels_doc))

        test = open(test_doc, "r")

        test_set = vectorizer.transform(test)
        #evaluate(read_all_labels(test_labels), clf.predict(test_set))
        print(clf.score(test_set, read_all_labels(test_labels)))

        #print(vectorizer.get_feature_names())



labels = read_all_labels_fine("TRAIN.txt")
#print(labels)
questions = pre_processing(read_questions("TRAIN.txt"))
#print(questions)
train = get_train_set(labels, questions)
#print(train)

SVM("TRAIN-questions.txt", "TRAIN-labels.txt", "DEV-questions.txt", "DEV-labels.txt", 1)



'''
        from sklearn.linear_model import SGDClassifier
        #clf = SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, max_iter = 100)
        clf = svm.SVC()
        clf.fit(X, read_all_labels(labels_doc))
        test = open(test_doc, "r")
        test_set = vectorizer.transform(test)
        evaluate(read_all_labels(test_labels), clf.predict(test_set))
        from sklearn.model_selection import GridSearchCV
        ##parameters_svm = {'vect__ngram_range': [(1, 1), (1, 2)], 'tfidf__use_idf': (True, False), 'clf-svm__alpha': (1e-2, 1e-3), }
        print(clf.get_params().keys())
        parameters_svm = {}
        gs_clf_svm = GridSearchCV(clf, parameters_svm, n_jobs=-1)
        gs_clf_svm = gs_clf_svm.fit(X, read_all_labels(labels_doc))
        #print("adeus")
        list = gs_clf_svm.predict(test_set)
        #print(list)
        #evaluate(read_all_labels(test_labels), list)
        #print("ola")
        print(gs_clf_svm.best_params_)
'''

'''
{'ngram_range': [(1, 1), (1, 2)], 'use_idf': (True, False)}
'''


#Não esquecer OutputCodeClassifier
#["as", "it", "these", "what", "a", "on", "``", "´´", "and", "to", ":", "`", "´"]