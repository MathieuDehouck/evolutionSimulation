import csv
from collections import defaultdict
from itertools import combinations
from os import scandir, listdir
from xml.etree import ElementTree

import numpy as np
import pandas as pd
from tqdm import tqdm

DATA_PATH = "../../Data/xmls"
SAVE_PATH = "coincide_mat.csv"


class WordSet:
    def __init__(self, file):
        self.file = file
        self.dict = {}
        self.populate()


    def populate(self):
        disco = ElementTree.parse(self.file).getroot()
        for cogset in disco:
            etym, reflexes = cogset[0], cogset[1]
            self.dict[etym[1].text] = {reflex[0].text: {reflex[1].tag: reflex[1].text} for reflex in reflexes}

    def __str__(self):
        return f"{self.file.name}:\n{self.dict}"


def save_mat(path, testeds=None):
    lang_dict = {}
    if testeds is None:
        testeds = []

    for flies in tqdm(scandir(DATA_PATH), total=len(listdir(DATA_PATH)), desc="Parsing"):
        if not flies.name.endswith(".xml") or flies.name == "Families.xml" or flies.name not in testeds: continue
        ws = WordSet(flies)
        for word in ws.dict:
            langs = ws.dict[word].keys()
            for l1, l2 in combinations(langs, 2):
                if l1 not in lang_dict:
                    lang_dict[l1] = {"word":0}
                if l2 not in lang_dict:
                    lang_dict[l2] = {"word":0}
                lang_dict[l1][l2] = lang_dict[l1].get(l2, 0) + 1
                lang_dict[l2][l1] = lang_dict[l2].get(l1, 0) + 1
                lang_dict[l1]["word"] += 1
                lang_dict[l2]["word"] += 1

    res_mat = np.zeros((len(lang_dict), len(lang_dict)))
    for (i, lang1), (j, lang2) in tqdm(combinations(enumerate(lang_dict), 2), total=int(len(lang_dict)*(len(lang_dict)-1)/2), desc="Populating Matrix"):
            res_mat[i][j] = lang_dict[lang1].get(lang2, 0)/lang_dict[lang1]["word"]
            res_mat[j][i] = lang_dict[lang2].get(lang1, 0)/lang_dict[lang2]["word"]

    lang_list = list(lang_dict.keys())

    with open(f"{DATA_PATH}/{path}", 'w') as f:
        csv.writer(f).writerow(["lang"] + lang_list)
        rows = res_mat.tolist()
        csv.writer(f).writerows([[lang_list[i]] + rows[i] for i in range(len(lang_list))])


def from_modern(moderns_targets, threshold=.05):
    res = defaultdict(lambda:0)
    for flies in scandir(DATA_PATH):
        if not flies.name.endswith(".xml") or flies.name == "Families.xml": continue
        ws = WordSet(flies)
        for word in ws.dict:
            langs = ws.dict[word].keys()
            if any(map(lambda x: x in moderns_targets, langs)):
                res[flies.name] += 1
    thresh = threshold * sum(res.values())
    res = dict(filter(lambda t: t[1] >= thresh, res.items()))
    res = list(res.keys())
    return res


def save_mat_with_targets(path, targets, testeds=None):
    lang_dict = defaultdict(lambda:{"word": 0})
    if testeds is None:
        testeds = []

    for flies in tqdm(scandir(DATA_PATH), total=len(listdir(DATA_PATH)), desc="Parsing"):
        if not flies.name.endswith(".xml") or flies.name == "Families.xml" or flies.name not in testeds: continue
        ws = WordSet(flies)
        for word in ws.dict:
            langs = ws.dict[word].keys()
            seen = defaultdict(lambda:True, zip(targets, (False for _ in targets)))
            for l1, l2 in combinations(langs, 2):
                if not seen[l1]:
                    lang_dict[l1]["word"] += 1
                    seen[l1] = True
                if not seen[l2]:
                    lang_dict[l2]["word"] += 1
                    seen[l2] = True
                if l1 not in targets or l2 not in targets:
                    continue
                if l1 not in lang_dict:
                    lang_dict[l1] = {
                        "word": 0}
                if l2 not in lang_dict:
                    lang_dict[l2] = {
                        "word": 0}
                lang_dict[l1][l2] = lang_dict[l1].get(l2, 0) + 1
                lang_dict[l2][l1] = lang_dict[l2].get(l1, 0) + 1

    lang_dict = dict(lang_dict)
    res_mat = np.zeros((len(lang_dict), len(lang_dict)))
    for (i, lang1), (j, lang2) in tqdm(
            combinations(enumerate(lang_dict), 2), total=int(len(lang_dict) * (len(lang_dict) - 1) / 2),
            desc="Populating Matrix"
            ):
        res_mat[i][j] = round(lang_dict[lang1].get(lang2, 0) / lang_dict[lang1]["word"], 3)
        res_mat[j][i] = round(lang_dict[lang2].get(lang1, 0) / lang_dict[lang2]["word"], 3)

    lang_list = list(lang_dict.keys())

    with open(f"{DATA_PATH}/{path}", 'w') as f:
        csv.writer(f).writerow(["lang"] + lang_list)
        rows = res_mat.tolist()
        csv.writer(f).writerows([[lang_list[i]] + rows[i] for i in range(len(lang_list))])


if __name__ == "__main__":
    moderns = ["French", "English", "Italian", "German", "Spanish", "Dutch", "Danish"]
    valids = from_modern(moderns)
    save_mat_with_targets("../west_europe_modern_mat.csv", targets=moderns, testeds=valids)
