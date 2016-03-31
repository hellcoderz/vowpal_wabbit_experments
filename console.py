import sys
from utils import examples, rmap, mapping, queries, SequenceLabeler, SequenceLabeler2, SequenceLabeler3, \
    SequenceLabeler4, SequenceLabeler3, chunk, preprocess, test, postprocess, query
import pyvw
from time import time
import os.path


def help():
    print """
    Commands:
    \t:c => initialize the model
    \t:t => train the model with the sample
    \t:q => query the model with the sample
    \t:s => save the model and quit
    """


def main():
    vw = []
    sl = []
    while True:
        inp = raw_input("> ")

        inp = inp.strip()
        words = inp.split()

        cmd = words[0]
        if cmd == "/save":
            for temp in vw:
                temp.finish()
            sys.exit(1)
        if cmd == "/train":
            data = " ".join(words[1:]).strip()
            for i in range(10):
                for temp in sl:
                    temp.learn(preprocess([data]))
        elif cmd == "/query":
            data = " ".join(words[1:]).strip()
            output = set()
            for s in sl:
                output.add(postprocess(query(s, data)))
            for out in output:
                print "\t", out
        elif cmd == "/start":
            data = " ".join(words[1:]).strip()
            if os.path.isfile(data + ".1") and os.path.isfile(data + ".2") and os.path.isfile(
                            data + ".3") and os.path.isfile(data + ".4"):
                vw = [
                    pyvw.vw("--quiet -i " + data + ".1 -f "+data + ".1"),
                    pyvw.vw("--quiet -i " + data + ".2 -f "+data + ".2"),
                    pyvw.vw("--quiet -i " + data + ".3 -f "+data + ".3"),
                    pyvw.vw("--quiet -i " + data + ".4 -f "+data + ".4")
                ]
            else:
                vw = [
                    pyvw.vw("--search 3 --quiet --search_task hook --ring_size 2048 -f " + data + ".1"),
                    pyvw.vw("--search 3 --quiet --search_task hook --ring_size 2048 -f " + data + ".2"),
                    pyvw.vw("--search 3 --quiet --search_task hook --ring_size 2048 -f " + data + ".3"),
                    pyvw.vw("--search 3 --quiet --search_task hook --ring_size 2048 -f " + data + ".4")
                ]
            sl = [
                vw[0].init_search_task(SequenceLabeler),
                vw[1].init_search_task(SequenceLabeler2),
                vw[2].init_search_task(SequenceLabeler3),
                vw[3].init_search_task(SequenceLabeler4)
            ]


if __name__ == "__main__":
    main()
