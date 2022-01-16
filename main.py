#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 11 21:26:35 2021

@author: masakieguchi

"""


import streamlit as st
from annotated_text import annotated_text
#from annotate_text import annotated_text

import pickle
import spacy
from ThemeAnalyzer import constituent_analysis, extract_theme, extract_theme_span, annotate, theme_markup_test



@st.cache(allow_output_mutation = True)
def load_model(spacy_model):
    nlp = spacy.load('spacy-model/en_core_web_md/en_core_web_md-2.3.1')
    #srl_pipe = SRLComponent(nlp, spacy_model)
    #nlp.add_pipe(srl_pipe, name='srl', last=True)
    return (nlp)

#Loading resources

def preprocess(text):
	# spaces
	text = text.strip()
	while "-\n" in text:
		text = text.replace("-\n", " ")
	while "\n" in text:
		text = text.replace("\n", " ")
	while "\t" in text:
		text = text.replace("\t", " ")
	while "  " in text:
		text = text.replace("  ", " ")
	while ";" in text:
		text = text.replace(";", ".")
	while ":" in text:
		text = text.replace(":", ".")
	return(text)
	
@st.cache(allow_output_mutation = True)
def list2dict(resource_list: list):
	dict_ = {}
	for l in resource_list.split('\n'):
		item = l.strip().split('\t')
		if len(item) < 1:
			continue
		else:
			dict_[item[0]] = item[1:]
	return(dict_)


resource = 'resources/'


#conjunctive = list2dict(pen(resource + 'conjunctive_adjuncts.txt', 'r').read())
#modal_adjunct = list2dict(open(resource + 'modal_adjuncts.txt', 'r').read())

#flatten = itertools.chain.from_iterable
#conjunctive_flat = list(flatten(conjunctive.values()))
#modal_flat = list(flatten(modal_adjunct.values()))

with open(resource + 'conjunctive_flat_clean_20210715.txt', 'r') as f:
	conjunctive_flat = list(f.read().split('\t'))

with open(resource + 'modal_adjuncts_flat_clean_20210715.txt', 'r') as f:
	modal_flat = list(f.read().split('\t'))


def theme_markup_test(text, nlp, conll=False, save = True):
	doc = nlp(text)
	const1 = constituent_analysis(doc)
	const1 = extract_theme(const1, doc, print_res=False)
	
	theme_span = extract_theme_span(const1) 
	
	anno_text = annotate(doc, theme_span, functions = colorfunctions)
	
	return (anno_text, const1)



### App layout
spacy_model = 'en_core_web_md'
nlp = load_model(spacy_model)

st.title("Theme analyzer—version 0.1 (beta)")

st.sidebar.title('Text to analyze')
colorfunctions = st.sidebar.checkbox('Color-code Theme functions')

text = st.sidebar.text_area("",height = 400)

rawresult = theme_markup_test(text, nlp, False, True)

st.header("Theme Annotation", "theme_annotation")
annotated_text(*rawresult[0])


st.header("Theme analysis", "theme_analysis")
st.write(rawresult[1])
