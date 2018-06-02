"""
Reads all relevant arff files, reduces the label space to the top k most
frequent labels and saves back the files. The data is saved into train/test
splits as well.
"""

import sys
import os
import luigi
import skml
# liac-arff
import arff

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lib.experimental_framework import load_from_arff

class PrepareData(luigi.Task):
    output_path = luigi.Parameter(default='data/')
    file_suffix = luigi.Parameter(default='-TOP-10')
    top_k = luigi.Parameter(default=10)

    def get_files(self):
        return {
            'scene': {
                'path': os.path.join(self.output_path,
                    'scene/scene.arff'),
                'labelcount': 6,
                'endian': "little"
                },
            'emotions': {
                'path': os.path.join(self.output_path,
                    'emotions/emotions.arff'),
                'labelcount': 6,
                'endian': "little",
            },
            'yeast-10': {
                'path': os.path.join(self.output_path,
                    'yeast/yeast.arff'),
                'labelcount': 14,
                'endian': "little"
            },
            'mediamill-10': {
                'path': os.path.join(self.output_path,
                    'mediamill/mediamill.arff'),
                'labelcount': 101,
                'endian': "little"
            },
            'enron-10': {
                    'path': os.path.join(self.output_path,
                        'enron/enron.arff'),
                'labelcount': 53,
                'endian': "little"
            },
            'medical-10': {
                    'path': os.path.join(self.output_path,
                        'medical/medical.arff'),
                'labelcount': 44,
                'endian': "little"
            },
            'slashdot-10': {
                    'path': os.path.join(self.output_path,
                        'slashdot/SLASHDOT-F.arff'),
                'labelcount': 22
            },
            'ohsumed-10': {
                    'path': os.path.join(self.output_path,
                        'ohsumed/OHSUMED-F.arff'),
                'labelcount': 23,
            },
            'tmc2007-500-10': {
                'path': os.path.join(self.output_path,
                    '../data/tmc2007-500/tmc2007-500.arff'),
                'labelcount': 22,
                'endian': "little"
            },
            'imdb-10': {
                    'path': os.path.join(self.output_path,
                        '../data/imdb/IMDB-F.arff'),
                    'labelcount': 28
            }
        }

    def _get_name(self, name):
        return os.path.join(self.output_path,
                    name.replace('-10', '').replace('-F', '')
                    + self.file_suffix + '.arff')


    def output(self):
        files = self.get_files()

        return {
            'local': [luigi.LocalTarget(self._get_name(f)) for f in files]
        }

    def run(self):
        out = self.output()
        k = int(self.top_k)
        files = self.get_files()

        for name in files:
            file_path = self._get_name(name)
            print(file_path)
            print("preparing: ", name)

            d = files[name]

            if 'endian' in d:
                data = load_from_arff(d['path'], labelcount=d['labelcount'],
                           return_attribute_definitions=True)
            else:
                data = load_from_arff(d['path'], labelcount=d['labelcount'],
                           return_attribute_definitions=True)

            X, y, a = data

            print("read!")

            N = X.shape[0]
            L = y.shape[1]
            print(N, L)

            split = False

            if L > 10:
                # FIXME: split
                pass

            if N < 10000:
                # 3-fold cross validation
                pass
            else:
                # 1/3 train-test split
                split = N // 3
                pass

            # FIXME: make @relation header to allow reading by meka
            rel_head = "@relation '{name}: -C {L}".format(name=name, L=L)

            # FIXME: get header

            if split:
                split = " -split-number {split}".format(split=split)
                rel_header += split

            rel_head += "'"


            # f = open(file_path, 'ab')
            # f.write(bytes((rel_head + '\n').encode('utf8')))
            # arff.dump(data, f)
