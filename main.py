#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 11 21:26:35 2021

@author: masakieguchi

"""

import streamlit as st
from annotated_text import annotated_text
#from annotate_text import annotated_text

import re
import pandas as pd
import pickle
import spacy
from ThemeAnalyzer import constituent_analysis, extract_theme, extract_theme_span, annotate, calc_measures, themeinfo_to_dict



@st.cache(allow_output_mutation = True)
def load_model(spacy_model):
	nlp = spacy.load('spacy-model/en_core_web_md/en_core_web_md-2.3.1')
	#srl_pipe = SRLComponent(nlp, spacy_model)
	#nlp.add_pipe(srl_pipe, name='srl', last=True)
	return (nlp)

#Loading resources
@st.cache(allow_output_mutation = True)
def preprocess(text):
	# spaces
	text = text.strip()
	text = re.sub("\n+", "\n", text)
	text = re.sub("(-\n)+", " ", text)
	text = re.sub("\t+", " ", text)
	text = re.sub("\s+", " ", text)
	#text = text.replace(";", ".")
	#text = text.replace(":", ".")
	#while "-\n" in text:
	#	text = text.replace("-\n", " ")
	#while "\t" in text:
	#	text = text.replace("\t", " ")
	#while "  " in text:
	#	text = text.replace("  ", " ")
	#while ";" in text:
	#	text = text.replace(";", ".")
	#while ":" in text:
	#	text = text.replace(":", ".")
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
	text = preprocess(text)
	doc = nlp(text)
	const1 = constituent_analysis(doc)
	const1 = extract_theme(const1, doc, print_res=False)
	
	theme_span = extract_theme_span(const1) 
	
	anno_text = annotate(doc, theme_span, functions = colorfunctions)
	
	return (anno_text, const1)



### App layout
spacy_model = 'en_core_web_md'
nlp = load_model(spacy_model)

st.title("Theme Analyzer—version 0.1 (beta)")

with st.expander("See explanation", False):
     st.write("""
        This is a demo version of Theme Analyzer. "Theme" as described in Systemic Functional Linguistics (Halliday & Matthiessen, 2014) is typically the first element(s) in a "clause", which serve(s) as a "point of departure" of the message.
		Effective choice of Theme is said to organize textual patterns, which may in turn help increase cohesion of the text. Theme Analyzer is an automatic approach to identify Theme and their grammatical realization details.
		Note that Theme Analyzer is still under development and its accuracy has only been tested on a small dataset of around 200 hand annotated sentences (although F1 of over .8).  
		Cite: Eguchi, M. (2021). Theme Analyzer demo. [Computer software]. 
     	""")

st.sidebar.title('Text to analyze')
colorfunctions = st.sidebar.checkbox('Color-code Theme functions')

text = st.sidebar.text_area("",height = 400)
st.sidebar.caption("""\nTheme Analyzer is still in development. This demo is to demonstrate an automatic approach to Theme Analysis.\n
                   """)
st.sidebar.markdown("Theme Analyzer is developed by [Masaki Eguchi](https://masakieguchi.weebly.com).")

rawresult = theme_markup_test(text, nlp, False, True)

st.header("Theme Annotation", "theme_annotation")

st.write(annotated_text(*rawresult[0]))


st.header("Theme Analysis", "theme_analysis")
df = pd.DataFrame.from_dict(themeinfo_to_dict(rawresult[1]), orient='index')
st.dataframe(df)

st.header("Detailed Grammatical Analysis", "details")
st.write(rawresult[1])

