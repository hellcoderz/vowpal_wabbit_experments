import sys
from utils import examples, rmap, mapping, queries, SequenceLabeler4, SequenceLabeler3, chunk, preprocess, test
import pyvw
from time import time
from flask import Flask, request
import json

app = Flask(__name__)

vw = None
sequenceLabeler = None


def train(dataset, fname="tagger.bin", N=10):
    # initialize VW as usual, but use 'hook' as the search_task
    try:
        # train it on the above dataset ten times; the my_dataset.__iter__ feeds into _run above
        start = time()
        load(fname)
        print '===============[training!]================'
        for i in xrange(N):
            sequenceLabeler.learn(dataset)
        end = time()
        print "TIME TAKEN:", (end - start) * 1000, "ms"
        vw.finish()
        load(fname)
        return True
    except:
        return False


def load(fname):
    global vw, sequenceLabeler
    try:
        vw = pyvw.vw("--quiet -i " + fname + " -f " + fname)
    except:
        vw = pyvw.vw("--search 3 --quiet --search_task hook --ring_size 2048 -f " + fname)
        vw.finish()
        vw = pyvw.vw("--quiet -i " + fname + " -f " + fname)
    sequenceLabeler = vw.init_search_task(SequenceLabeler3)


@app.route('/train')
def trainAPI():
    global vw, sequenceLabeler
    model = request.args.get("model", "tagger1.bin")
    N = request.args.get("iter", 10)
    try:
        data = request.args.get("data")
        data = data.strip()
        print "Traning:", data
        sequenceLabeler.learn(preprocess([data]))
        return "model trained"
    except:
        return "'data' field not present OR trainning error!!!"


@app.route('/test')
def testAPI():
    global vw, sequenceLabeler
    model = request.args.get("model", "tagger1.bin")
    try:
        query = request.args.get("query")
        query = query.strip().lower()
        chunked = test(sequenceLabeler, query)
        return json.dumps(chunked)
    except:
        return "'query' field not present OR error in prediction"


if __name__ == "__main__":
    load("tagger1.bin")
    app.debug = True
    app.run(host='0.0.0.0')

    # load("tagger1.bin")
    #
    # train(preprocess(["harry_B potter_I"]), "tagger1.bin")
    # load("tagger1.bin")
    # test(sequenceLabeler, "watch harry potter")
    #
    # train(preprocess(["watch_N big_B bang_I"]), "tagger1.bin")
    # load("tagger1.bin")
    # test(sequenceLabeler, "watch harry potter")
