import sys
from sklearn import svm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multiclass import OneVsRestClassifier
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer

class Tokenizer:
    def __init__(self):
        self.wnl = WordNetLemmatizer()
        self.ps = PorterStemmer()
    def __call__(self, doc):
        return [self.ps.stem(o) for o in [self.wnl.lemmatize(t) for t in word_tokenize(doc)]]

def read_labels(doc, fine):

    file = open(doc, "r")
    labels = []
    for sentence in file:
        split_character = " " if fine else ":"
        label = sentence.split(split_character)[0]

        if fine:
            labels += [label.split("\n")[0]]
        else:
            labels += [label]

    return labels

def read_train_questions(doc):

    file = open(doc, "r")
    question_list = []
    for sentence in file:
        question_list += [sentence.split(" ", 1)[1].split("\n")[0]]

    return question_list

def read_test_questions(doc):

    file = open(doc, "r")
    question_list = []
    for sentence in file:
        question_list += [sentence.split("\n")[0]]

    return question_list

def SVM_coarse(labels, questions, test_doc):

    stop_words = ["other", "thi", "but", "'s", "if", "by", "as", "it", "these", "what", "a", "on", "``", "´´", "and", "to", ":", "`", "´"]
    vectorizer = TfidfVectorizer(strip_accents = 'unicode', stop_words = stop_words, min_df = 3, tokenizer=Tokenizer())
    X = vectorizer.fit_transform(questions)

    clf = svm.SVC()
    clf.fit(X, labels)

    test_set = vectorizer.transform(open(test_doc, "r"))

    return clf.predict(test_set)

def SVM_fine(labels_coarse, labels_fine, questions, test_doc):

    coarse_classifications = SVM_coarse(labels_coarse, questions, test_doc)
    stop_words = ["'", "'t", ',', '-', '.', '..', "?", "'s", "as", "it", "these", "what", "a", "an", "``", "´´", "and", "to", ":", "`", "´"]

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

    for i in range(len(labels_coarse)):
        if labels_coarse[i] == "LOC":
            QUESTIONS_LOC += [questions[i]]
            Y_LOC += [labels_fine[i]]
        elif labels_coarse[i] == "ENTY":
            QUESTIONS_ENTY += [questions[i]]
            Y_ENTY += [labels_fine[i]]
        elif labels_coarse[i] == "DESC":
            QUESTIONS_DESC += [questions[i]]
            Y_DESC += [labels_fine[i]]
        elif labels_coarse[i] == "HUM":
            QUESTIONS_HUM += [questions[i]]
            Y_HUM += [labels_fine[i]]
        elif labels_coarse[i] == "NUM":
            QUESTIONS_NUM += [questions[i]]
            Y_NUM += [labels_fine[i]]
        else:
            QUESTIONS_ABBR += [questions[i]]
            Y_ABBR += [labels_fine[i]]

    vectorizer_X_LOC = TfidfVectorizer(strip_accents = 'unicode', stop_words = stop_words, min_df = 3, tokenizer=Tokenizer())
    X_LOC = vectorizer_X_LOC.fit_transform(QUESTIONS_LOC)
    vectorizer_X_ENTY = TfidfVectorizer(strip_accents = 'unicode', stop_words = stop_words, min_df = 3, tokenizer=Tokenizer())
    X_ENTY = vectorizer_X_ENTY.fit_transform(QUESTIONS_ENTY)
    vectorizer_X_DESC = TfidfVectorizer(strip_accents = 'unicode', min_df = 3, tokenizer=Tokenizer())
    X_DESC = vectorizer_X_DESC.fit_transform(QUESTIONS_DESC)
    vectorizer_X_HUM = TfidfVectorizer(strip_accents = 'unicode', stop_words = stop_words, min_df = 3, tokenizer=Tokenizer())
    X_HUM = vectorizer_X_HUM.fit_transform(QUESTIONS_HUM)
    vectorizer_X_NUM = TfidfVectorizer(strip_accents = 'unicode', min_df = 3, tokenizer=Tokenizer())
    X_NUM = vectorizer_X_NUM.fit_transform(QUESTIONS_NUM)
    vectorizer_X_ABBR = TfidfVectorizer(strip_accents = 'unicode', min_df = 3, tokenizer=Tokenizer())
    X_ABBR = vectorizer_X_ABBR.fit_transform(QUESTIONS_ABBR)

    clf_LOC = OneVsRestClassifier(svm.SVC())
    clf_LOC.fit(X_LOC, Y_LOC)
    clf_ENTY = OneVsRestClassifier(svm.SVC())
    clf_ENTY.fit(X_ENTY, Y_ENTY)
    clf_DESC = OneVsRestClassifier(svm.SVC())
    clf_DESC.fit(X_DESC, Y_DESC)
    clf_HUM = OneVsRestClassifier(svm.SVC())
    clf_HUM.fit(X_HUM, Y_HUM)
    clf_NUM = svm.SVC()
    clf_NUM.fit(X_NUM, Y_NUM)
    clf_ABBR = svm.SVC()
    clf_ABBR.fit(X_ABBR, Y_ABBR)

    predicted_labels = []
    test_questions = read_test_questions(test_doc)

    for i in range(len(coarse_classifications)):
        if coarse_classifications[i] == "LOC":
            predicted_labels += [clf_LOC.predict(vectorizer_X_LOC.transform([test_questions[i]]))]
        elif coarse_classifications[i] == "ENTY":
            predicted_labels += [clf_ENTY.predict(vectorizer_X_ENTY.transform([test_questions[i]]))]
        elif coarse_classifications[i] == "DESC":
            predicted_labels += [clf_DESC.predict(vectorizer_X_DESC.transform([test_questions[i]]))]
        elif coarse_classifications[i] == "HUM":
            predicted_labels += [clf_HUM.predict(vectorizer_X_HUM.transform([test_questions[i]]))]
        elif coarse_classifications[i] == "NUM":
            predicted_labels += [clf_NUM.predict(vectorizer_X_NUM.transform([test_questions[i]]))]
        else:
            predicted_labels += [clf_ABBR.predict(vectorizer_X_ABBR.transform([test_questions[i]]))]

    return predicted_labels

args = sys.argv

predicted_labels = []
if args[1] == "-coarse":
    labels = read_labels(args[2], 0)
    questions = read_train_questions(args[2])
    predicted_labels = SVM_coarse(labels, questions, args[3])

elif args[1] == "-fine":
    labels_coarse = read_labels(args[2], 0)
    labels_fine = read_labels(args[2], 1)
    questions = read_train_questions(args[2])
    predicted_labels = SVM_fine(labels_coarse, labels_fine, questions, args[3])

for label in predicted_labels:
    if args[1] == "-fine":
        print(label[0])
    else:
        print(label)