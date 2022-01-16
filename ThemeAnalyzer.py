#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 11 21:26:35 2021

@author: masakieguchi

Requires Spacy v2 for Transformer-SRL
On MBP use conda activate tf-srl


On M1 mac use conda activate nlp
- On M1 mac, tesorflow should be completed from the source. No Rosetta 2 distribution.
https://github.com/tensorflow/tensorflow/issues/46044
- Then I installed from the wheel, which updated numpy version.
- The numpy was not compatible with benepar,
- I reinstalled pip install -U numpy==numpy-1.21.0 and prioritized benepar component.
- THis was successful.


This program is an emulation of the ThemeAnalyzer by Park and Lu (2015).

Park, K., & Lu, X. (2015). Automatic analysis of thematic structure in
written English. International Journal of Corpus Linguistics,
20(1), 81–101. https://doi.org/10.1075/ijcl.20.1.04par

Penn Treebank II Constituent Tags
http://www.surdeanu.info/mihai/teaching/ista555-fall13/readings/PennTreebankConstituents.html


Downloaded transformer_srl from pip
- The benepar version is downloaded from github (version 0.1.3)
- this solved the dependency issue.
- Then I ran setup.py for the benepar git (self-attentive-parser)


Ver 0.2:
	Added semantic role labeling option through Transformer SRL.
	- compatible with SpaCy 2.3

Ver 0.3: (2021.06.25)
	- Added list match for preliminary Conjunctive and modal adjunct list.

Migrated to SFL analyzer

Ver 0.2:
	- changed the behavior of constituency parsing.
	- Theme extraction adding marked , predicated tags in result

Ver 0.3 (2021.07.06)
	- Refactor predicated_identification into special_theme_identification

Ver 0.4
	- refactor theme extraction pipeline
	- dropped benepar from the component
"""


# Loading packages
#from bs4 import BeautifulSoup as bs
#import lxml
from collections import OrderedDict

import copy
import csv
import glob
import itertools
import json

import pickle
import pprint as pp
from operator import itemgetter, attrgetter
import re
from setuptools import setup, find_packages

#import explacy
import spacy

#import benepar
#from benepar.spacy_plugin import BeneparComponent

from spacy.language import Language
from spacy.lang.en import English
from spacy.tokens import Doc, Token
# from spacy.lang.en import tag_map


from collections import Counter
import math

import os.path
from os import path

#
import streamlit as st
#from annotated_text import annotated_text
from annotate_text import annotated_text

#Constructing SpaCy pipeline


@st.cache(allow_output_mutation = True)
def load_model(spacy_model):
    nlp = spacy.load('en_core_web_md')
    #srl_pipe = SRLComponent(nlp, spacy_model)
    #nlp.add_pipe(srl_pipe, name='srl', last=True)
    return (nlp)


#Loading resources

@st.cache
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




# Verb Atlas
with open(resource + 'pb2va_integrated.pickle', 'rb') as f:
	pb2va = pickle.load(f)

# VerbNet 3.3
with open(resource + 'pb2vn_20210702.pkl', 'rb') as f:
	pb2vn = pickle.load(f)


#functions

def parse_print(text):
	print("\t".join(["Token", "lemma", "PennTag", "UPOS", "DepType", "Head"]))
	txt = nlp(text)
	print(list(txt.noun_chunks) + list(txt.ents) +
	      list([x.label_ for x in txt.ents]))
	for sent in txt.sents:

		for token in txt:
			print("\t".join([token.text, token.lemma_, token.tag_,
			      token.pos_, token.dep_, token.head.text]))


#Modified mood analysis
def has_wh_element(sent, aux):
	# "What is your name?"
	wh_tags = ["WDT", "WP", "WP$", "WRB"]
	wh_words = [t for t in sent if t.tag_ in wh_tags]
	
	if not wh_words:
		return False
	
	has_wh_before_aux = wh_words[0].i < aux.i
	
	#print(wh_words)
	# Include pied-piped constructions: "To whom did she read the article?"
	pied_piped = (wh_words and wh_words[0].head.dep_ == "prep") and wh_words[0].i < aux.i

	# Exclude pseudoclefts: "What you say is impossible."
	pseudocleft = wh_words and wh_words[0].head.dep_ in ["csubj", "advcl"]
	if pseudocleft:
		return False

	return (has_wh_before_aux or pied_piped)

def _is_subject(tok):
	'''
	from
	https://towardsdatascience.com/linguistic-rule-writing-for-nlp-ml-64d9af824ee8
	'''
	subject_deps = {"csubj", "nsubj", "nsubjpass"}
	return tok.dep_ in subject_deps


def is_question(sent):
	''' Return Booleans whether the sent is a question and has wh-element in the question
	Based on the code from 
	https://towardsdatascience.com/linguistic-rule-writing-for-nlp-ml-64d9af824ee8
	
	# return Root and subject
	
	'''
	# Setting the defalt parameter to be False
	is_question = False
	has_wh = False
	
	# retrieve the root.
	root = [t for t in sent if t.dep_ == "ROOT"][0]  # every spaCy parse as a root token!
	subj = [t for t in root.children if _is_subject(t)]

	# Type I: In a non-copular sentence, "is" is an aux.
	# "Is she using spaCy?" or "Can you read that article?"
	aux = [t for t in root.lefts if t.dep_ in ["aux", "auxpass"]]
	if subj and aux:
		if aux[0].i < subj[0].i:
			is_question = True

	# Type II: In a copular sentence, "is" is the main verb.
	# "Is the mouse dead?"
	root_is_inflected_copula = (root.pos_ == "AUX" and root.tag_ != "VB")
	if subj and root_is_inflected_copula:
		if root.i < subj[0].i:
			is_question = True
			aux = [root] # if the root is aux, update the aux
		
	## Sometimes the sentense have two attr not a subject
	attr = [t for t in root.children if t.dep_ == "attr"]
	#print(attr)
	if len(attr) > 1 and root_is_inflected_copula:
		if root.i < attr[0].i:
			is_question = True
			aux = [root] # if the root is aux, update the aux
			subj = [attr[0]] # I will treat the first attribute as subject
	
	# if the sentence is question search for Wh-element
	if is_question:
		if has_wh_element(sent, aux[0]):
			has_wh = True
	
	if is_question: #if this is a question, we need to tag aux and subject as  they are potential theme
		return (is_question, has_wh, [aux[0], subj[0]])
	else:
		return (is_question, has_wh, (None, None))


def test_interrogative(text, parse_print = True):
	doc = nlp(text)
	#print(constituent_analysis(doc))
	for sent in doc.sents:
		if parse_print == True:
			try:
				print(text)
				#explacy.print_parse_info(nlp, sent.text)
			except AssertionError:
				print('parse error')
		#print(sent)
		return(is_question(sent))
	
#test_interrogative("Are they still together?")
def is_imparative(sent):
	
	is_imparative = False
	### Note: Vocatives may be confused with subjects.
	### When comma is present between the NP and root, the NP is npadvmpd
	subj = ['nsubj', 'csubj', 'nsubjpass', 'csubjpass', "acomp", "prep", "attr", "expl"]
	
	
	root = [t for t in sent if t.dep_ == "ROOT"] # did not slice because list is the desirable type
	left_dep = [l.dep_ for l in  root[0].lefts] #
	
	if not any(dep in subj for dep in left_dep):
		mood = "Imparative"
		is_imparative = True
	
	
	if is_imparative and root[0].text.lower() in ['let']:
		try:
			print(sent.text)
			#explacy.print_parse_info(nlp, sent.text)
		except AssertionError:
			print('parse error')
			print(sent)
			print([w.dep_ for w in sent])
		child = [w for w in root[0].children if w.dep_ in ['ccomp', 'xcomp']] #main verb
		if len(child) > 0:
			child2 = [w2 for w2 in child[0].children if 'subj' in w2.dep_]
			#print(child2)
			root.extend(child2)
		

	return (is_imparative, root)

def test_imparative(text, parse_print = True):
	doc = nlp(text)
	for sent in doc.sents:
		if parse_print == True:
			try:
				explacy.print_parse_info(nlp, sent.text)
			except AssertionError:
				print('parse error')
		print(sent)
		return(is_imparative(sent))
	
# test_imparative("Have you finished your meal sir?")
# test_imparative("Do the dishes?")
# test_imparative("Don't go away.")
# test_imparative("Answer no more than three of the following questions.")
#test_imparative("You, listen to me, young man.")
# 
# =============================================================================

# =============================================================================
# def has_finite(sent):
# 	finite = False
# 	root = [w for w in sent if w.dep_ == "ROOT"]
# 	
# 	if root.pos_ in ['VERB', "AUX"]:
# 		if root.tag_ in ["VBZ", "VBP", "VBD"]:
# 			finite =  True
# 		elif root.tag_ in ["VBG", "VBN"]:
# 			aux = [w for w in root.children if w.dep_ == 'aux']
# 			if any(tag in ['MD', "VBZ", "VBP", "VBD"])
# =============================================================================

def mood_analysis_v2(sent):
	'''
	
	Parameters
	----------
	sent : spaCy sent object
		DESCRIPTION.

	Returns
	-------
	Results for mood category
	

	'''
	interrogative, wh_question, aux_subj = is_question(sent)
	imparative, root = is_imparative(sent)
	
	if interrogative and  wh_question:
		return ("Wh- Interrogative", None)
	elif interrogative:
		return ("Yes/No Interrogative", aux_subj)
	elif imparative:
		return ("Imperative", root)
	else:
		return("Declarative", None)


def srl_list_refiner(srl_tags: list) -> list:
	'''

	Parameters
	----------
	srl_tags : list
		List returned by srl tags.

	Returns
	-------
	list
		refined srl role list.

	'''
	holder = []
	for x in srl_tags:
		match = re.match(r'B-ARG(\d)', x)  # match the numbered argument
		if match:
			x = "A" + match.group(1)
			holder.append(x)
	return holder

@st.cache
def safe_devide(num, denom):
	if denom == 0:
		return 0
	else:
		return num / denom

@st.cache
def counter_cosine_similarity(c1, c2):
	'''
	taken from  stackoverflow
	https://stackoverflow.com/questions/14720324/compute-the-similarity-between-two-lists/14720386
	requires math
	'''
	terms = set(c1).union(c2)
	dotprod = sum(c1.get(k, 0) * c2.get(k, 0) for k in terms)
	magA = math.sqrt(sum(c1.get(k, 0)**2 for k in terms))
	magB = math.sqrt(sum(c2.get(k, 0)**2 for k in terms))
	return safe_devide(dotprod,  (magA * magB))

@st.cache
def length_similarity(c1, c2):
    lenc1 = sum(c1.values())
    lenc2 = sum(c2.values())
    return min(lenc1, lenc2) / float(max(lenc1, lenc2))

def compute_similarity_from_lists(listA: list, listB: list, funct='length'):
	'''
	Parameters
	----------
	listA : list
		list of string elements
	listB : list
		list of string elements
	funct : TYPE, optional
		length measure or pure cosine measure

	Returns
	-------
	score : TYPE
		cosine similarity (either lengths adjusted).
	'''

	counterA = Counter(listA)
	counterB = Counter(listB)

	if len(counterA) < 1:
		return 0
	if funct == 'length':
		score = length_similarity(counterA, counterB) * \
		                          counter_cosine_similarity(counterA, counterB)
	else:
		score = counter_cosine_similarity(counterA, counterB)

	return score

def retrieve_predicate(pb_predicate, roles, resource='va'):
	'''
	Parameters
	----------
	pb_predicate : TYPE
		String of propbank predicate id (e.g., take.01).
	roles : TYPE
		list of semantic roles e.g., A0, A1
	resource : TYPE, optional
		what predicate resource to use; currently verbatlas or verbNet

	Returns
	-------
	check : Booleen
		Whether the predicate id is matched with the resource.
	res : DIct
		dictionary with predicate and semantic role information.

	'''
	if resource == 'va':
		try:
			res = pb2va[pb_predicate]
			check = True
		except KeyError:
			res = {'frameid': None,
				    'roles': roles,
					'predicate': pb_predicate,
					'sel_pref': None,
					}
			check = False

		return (check, res)

	elif resource == 'vn':
		try:
			candidate = pb2vn[pb_predicate]['entries']
			scores = []

			for idx, info in enumerate(candidate):

				role_list = info['roles'].keys()
				# print('role list' + str(role_list))
				score = compute_similarity_from_lists(roles, role_list, 'length')
				scores.append(score)  # similarity between lists

			# print(scores)

			maxid = scores.index(max(scores))
			res = candidate[maxid]
			check = True

		except KeyError:
			res = {'frameid': None,
				    'roles': roles,
					'frame': pb_predicate,
					'sel_pref': None}
			check = False
		return (check, res)

def identify_specialtheme(token):
	'''Returns the information necessary toextract

	Parameters
	----------
	token : spacy Token object
		DESCRIPTION.

	Returns
	-------
	res : dictionary
		DESCRIPTION.

	'''

	res = {"value": False,
		  "type": None,
		  "structure": [],
		  'headids': {'it_subj': None,
					  'predicated': None,
					  'Rheme': None,
					  'Rheme_left': None,
					  'acomp': None,
					  'ccomp': None,
					  'xcomp': None,
					  'modal': None,
					  'auxpass': None,
					  'verbpp': None,
						},
		  'otherheads': [],
		  }
	l = []
	r = []

	# enter special theme detection if the main verb is be
	if token.lemma_ in ["be"]:

		# Find structural info while iterating the children
		for w in token.children:

			if w.dep_ == 'nsubj' and w.text.lower() == 'it':
				# "it" should be the subject
				res['headids']['it_subj'] = w.i

			elif w.dep_ in ['attr', 'prep'] or (w.dep_ in ['advmod', 'npadvmod'] and (w.i > token.i)):
				# predication here
				res['headids']['predicated'] = w.i

				# if the predicated part is NP, dep parsing returns relcl
				# in this case relcl should be in the Rheme position
				for w2 in w.children:  # relative clauses are trickyy
					if w2.dep_ == 'relcl':
						res['headids']['Rheme'] = w2.i
						res['headids']['Rheme_left'] = w2.left_edge.i
						r.append(w2)
						
						to_in_Rheme = [c for c in w2.children if (c.dep_ in ["aux"] and c.text == 'to')]
						
						if to_in_Rheme:
							res['headids']['xcomp'] = w.left_edge.i
						else:
							res['headids']['ccomp'] = w.left_edge.i



			# For Thematized comment
			elif w.dep_ in ['acomp']:
				res['headids']['acomp'] = w.i
				
			# for thematized comment
			elif w.dep_ in ['aux'] and w.tag_ in ['MD']:
				#print(w.text)
				res['headids']['modal'] = w.i

			# The last piece is the ccomp as rheme
			elif w.dep_ in ['ccomp'] and token.i < w.i:
				res['headids']['Rheme'] = w.i
				res['headids']['Rheme_left'] = w.left_edge.i
				res['headids']['ccomp'] = w.left_edge.i
				
			
			elif w.dep_ in ['xcomp'] and token.i < w.i:
				res['headids']['Rheme'] = w.i
				res['headids']['Rheme_left'] = w.left_edge.i
				res['headids']['xcomp'] = w.left_edge.i
				
			else:
				res['otherheads'].append(w.i)
	
	#other thematized comment:
		#it is regretted that
		
	elif token.tag_ in ["VBD", "VBN"] and token.lemma_ not in ["be"]:
		res['headids']['verbpp'] = token.i
		
		for w in token.children:
			
			if w.dep_ in ['nsubjpass', 'nsubj'] and w.text.lower() == 'it':
				# "it" should be the subject
				res['headids']['it_subj'] = w.i

			# for thematized comment
			elif w.dep_ in ['aux'] and w.tag_ in ['MD']:
				#print(w.text)
				res['headids']['modal'] = w.i

			elif w.dep_ in ['auxpass']:
				#print(w.text)
				res['headids']['auxpass'] = w.i

			# For Thematized comment
			elif w.dep_ in ['acomp']:
				res['headids']['acomp'] = w.i

			# The last piece is the ccomp as rheme
			elif w.dep_ in ['ccomp'] and token.i < w.i:
				res['headids']['Rheme'] = w.i
				res['headids']['Rheme_left'] = w.left_edge.i
				res['headids']['ccomp'] = w.left_edge.i
			else:
				res['otherheads'].append(w.i)
	# it subj, acomp and xcomp - It is difficult to 
		
	
	## mental verb e.g., I think that he + is a genious.
	
	
	else:
		return (res, l, r)
	
	# Predicate theme
	if None not in [res['headids']['it_subj'], res['headids']['predicated'], res['headids']['Rheme'], res['headids']['ccomp']]:
		res['value'] = True
		res['type'] = "predicated"
		# once theme-rheme boundary is detected, store the info
		l.extend([(word, word.i) for word in token.children if word.i < res['headids']['Rheme_left']])
		l.append((token, token.i))
		#print("In predicate branch" + str(l))
		l = sorted(l, key=itemgetter(1))
		l = [w[0] for w in l]
		
		r.extend([word for word in token.children if word.i >=
		                res['headids']['Rheme_left']])

	# Thematized comment theme ## Interpersonal projection e.g., It is true that
	# 1) it as subject, acomp, and ccomp as condition
	## setting the boundary for Theme Rheme distinction
	elif None not in [res['headids']['it_subj'], res['headids']['acomp'], res['headids']['Rheme']] or \
		None not in [res['headids']['it_subj'], res['headids']['auxpass'], res['headids']['verbpp'], res['headids']['Rheme']] or \
		None not in [res['headids']['it_subj'], res['headids']['modal'], res['headids']['Rheme']] and \
		(None in [res['headids']['acomp'], res['headids']['predicated']]):
		res['value'] = True
		res['type'] = "thematized comment"
		# once theme-rheme boundary is detected, store the info
		
		#res['l'].extend([(word, word.i) for word in token.subtree if word.i < res['headids']['Rheme']])
		boundary = "Rheme"
		## without subtree
		for word in token.children:
			
			if word.i < res['headids']['Rheme']:
				l.extend([(word, word.i)])
			#Theme Rheme boundary is different set theme rheme boundary
			elif word.dep_ in ["ccomp"] and word.i > token.i: 
				l.extend([(sub, sub.i) for sub in word.children if sub.i < res['headids']['Rheme']])
				
			elif word.dep_ in ['xcomp']  and word.i > token.i: #but only when the 
				boundary = 'Rheme_left'
				l.extend([(sub, sub.i) for sub in word.children if sub.i < res['headids']['Rheme_left']])
				
		l.append((token, token.i))
		print("In thematized comment branch" + str(l))
		l = sorted(l, key=itemgetter(1))
		l = [w[0] for w in l]
		r.extend([word for word in token.children if word.i >= res['headids'][boundary]]) #may be a 

	
	## Mental or Verbal 
	#print(res)
	return(res, l, r)


def specialtheme_test(text):
	doc = nlp(text)
	#explacy.print_parse_info(nlp, doc.text)

	for token in doc:
		if token.dep_ in ["ROOT", 'conj'] and token.pos_ in ["VERB", "AUX"]:
			identify_specialtheme(token)


# =============================================================================
# test("It was a long and hard-fought campaign that was ended by a 5 - 4 decision in the Supreme Court to halt the counting of votes in the key state of Florida.")
# 
# test("It was a long and hard-fought campaign that ended in the key state of Florida.")
# specialtheme_test("It was a long and hard-fought campaign that we ended in the key state of Florida.")
# specialtheme_test("It's a brutal, unforgiving place in which love outside the norm struggles to be something more than a self-destructive gesture.")
# 
# test("it is time to boast about the safety record of the chemical industry.")
# test("It was my wife, Tipper, who first suggested that I put together a book with pictures and graphics to make the whole message easier to follow, combining many elements from my slide show with all of the new original material I have compiled over the last few years.")
# test('I also want to convey my strong feeling that what we are facing is not just a cause for alarm, it is paradoxically also a cause for hope.')
# 
# =============================================================================

def coordinated_clause_category(v_token, mood_type):
	
	if v_token.dep_ == "conj" and len([t for t in v_token.children if _is_subject(t)]) == 0:
		#print("Should be coordinated phrases")
		if mood_type == "Imperative":
			cl_status = "Coordinated clause"
		else:
			cl_status = "Coordinated VP"
	
	elif v_token.dep_ == "conj" and len([t for t in v_token.children if _is_subject(t)]) > 0:
		print("coordinated clauses with explicit subjects")
		cl_status = "Coordinated clause"
	
	else:
		cl_status = v_token.dep_
	
	return(cl_status)

def constituent_analysis(doc, parse_print=False):
	'''Parses spacy Doc object and parse into constituencies

	Parameters
	----------
	doc : spacy doc object
	parse_print : Boolean
		DESCRIPTION. The default is True.

	Returns
	-------
	None.


		
	'''
	clauses = {} #dictionary where main verbs of a T-unit as key
	
	for sentid, sent in enumerate(doc.sents):
		n_of_mainverbs = 0
		
		if parse_print == True:
			try:
				print(sent.text)
				#explacy.print_parse_info(nlp, sent.text)
			except AssertionError:
				print('parse error')
			finally:
				#print('Constituency:')
				#print(sent._.parse_string)
				
				#semantic role deactivated
				#print('Semantic roles:') #semantic role deactivated
				#pp.pprint(doc._.srl[sentid], sort_dicts=False)
				print("---\n")

		# mood_analysis_v2 uses dependency
		mood_type, potential_theme = mood_analysis_v2(sent) # maybe incorporate imparative here?? 

		for token in sent:
			# 1) T-unit detection, by locating main verbs

			# 2) Coordinated clauses are separated and analyzed separately.
				# -> It may be better to split the coordinated clauses/
				### Subordinate clauses are Theme it they precedes the main clause ###
				# They are simply Rheme when they follow the main clause. No theme is identified within the subordinate clause.

			# When token is not the root or the head of coordinated elements, skip them
			# Enter constituency analysis if the token is ROOT or the head of coodinated elements
			#Check if the token is mainverb.
			
			_is_mainverb = token.dep_ in ["ROOT"] and token.pos_ in ["VERB", "AUX"]
			_is_conjuncted_clause = token.dep_ in ["conj"] and token.head.dep_ in ["ROOT"] and token.pos_ in ["VERB", "AUX"]
			
			if _is_mainverb or _is_conjuncted_clause:
				#print(sent)
				
				## differentiate, coordinated VP, Coordinated clause
				## Note if the clause is imperative, clause coordination lacks subject,
				## which is handled in this function
				cl_status = coordinated_clause_category(token, mood_type)
				
				if cl_status == "Coordinated VP": # avoid extracting theme from coordinated VP
					continue 
				
				# identify special theme; this is necessary to check predicated theme
				specialtheme, l, r = identify_specialtheme(token)
				
				# semantic role labeling
				
# =============================================================================
# 				if len(doc._.srl[sentid]['verbs']) == 0:
# 					srls = []
# 					predicate = None
# 					pred_score = 0
# 				else:
# 					## mapping Spacy token id and Transformer-srkl is still experimental
# 					try:
# 						srls = [verb['tags'] for verb in doc._.srl[sentid]['verbs'] if verb['verb'] == token.text][0]
# 						predicate = [verb['frame'] for verb in doc._.srl[sentid]['verbs'] if verb['verb'] == token.text][0]
# 						pred_score = [verb['frame_scores'] for verb in doc._.srl[sentid]['verbs'] if verb['verb'] == token.text][0]
# 					except IndexError:
# 						srls = [verb['tags'] for verb in doc._.srl[sentid]['verbs']][0]
# 						predicate = [verb['frame'] for verb in doc._.srl[sentid]['verbs']][0]
# 						pred_score = [verb['frame_scores'] for verb in doc._.srl[sentid]['verbs']][0]
# 	
# 					refined_srl = srl_list_refiner(srls)
# 	
# 				if pred_score > .7:
# 					pred_frame = retrieve_predicate(predicate, refined_srl, resource='vn')
# 				else:
# 					pred_frame = None
# 
# =============================================================================
				# Holder for head
				heads = {'l': [],  # left
						 'r': []}  # right

				# Added step: Store heads of each constituency for subtree analysis
				# - left, main verb group or right.
				# - Main verb depricated because of necessary to include them in left.
				# print(info_predicated['value'])
				if mood_type in ['Yes/No Interrogative', 'Imperative']:
					heads['l'].extend([left for left in token.lefts])
					if token.dep_ not in ['conj']:
						heads['l'].extend(potential_theme) #potential theme is in shape of (aux, nsubj)
					elif token.dep_ in ['conj']:
						heads['l'].append(token)
					
					for right in token.rights: #the remaining heads should be in rights
						if right not in heads['l']:
							heads['r'].append(right)
				
				
				elif not specialtheme['value']:
					heads['l'] = [left for left in token.lefts]
					#heads['l'].append(token)
					# heads['mv'] = [left  for left in token.lefts if left.dep_ in ['aux', 'auxpass', 'neg']]
					# another option if (right.dep_ not in ['conj'] and right.pos_ not in ['VERB', "AUX"])
					
					
					for right in token.rights:			### This chunk is rewritten to handle fronted WH elements from subtree
						if right.dep_ not in ['ccomp', 'xcomp', 'acl']: # when no clausal element, 
							heads['r'].append(right) #just store them as rights
							
						else:
							for sub_l in right.lefts: #subtree in left should be compared with the id of main verb
								if sub_l.i < token.i: # if subtree is left than mainverb
									heads['l'].append(sub_l)
								elif sub_l.i > token.i: #if subtree is right than mainverb
									heads['r'].append(sub_l)
							#heads['r'].append(right) # rights are rights/
							heads['r'].extend([sub_r for sub_r in right.rights]) # rights are rights/
				
					
				else:
					heads['l'] = l
					heads['r'] = r
				
				
				#print(heads)
				# The extract subtree function for each main verb extract detailed constituency info.
				# note this is nested in dictionary
				n_of_mainverbs += 1 #record how many main verbs were stored
				clauses[token.i] = {'sentence': sent.text,
									'sentid': str(sentid),
									'lexverb': token.text,
									'cl.status': cl_status,
									#'transitivity': {'propbank': predicate,
									#				 'pred_score': pred_score,
									#				 'frame_roles': pred_frame,
									#				 'srl_tags': srls,
									#				  },
									'theme': {'text': [],
											  'headids': [],
											  'lengths': 0,
											  'functions': [],
											  'roles': [],
											  'theme_type': None, #unmarked/ marked/enhanced
											  'sub_type': None,
											  #'marked': False,
											  #'marked_role': None,
											  #'equative': False,
											  'mood_type': mood_type,
											  'special_theme': specialtheme,
											  "type": specialtheme["type"]
											  },
									'constit': extract_subtrees(heads, doc, sent, sentid, token, specialtheme, cl_status)}
		
		if n_of_mainverbs ==0:
			root = [w for w in sent if w.dep_ == "ROOT"]
			clauses[token.i] = {'sentence': sent.text,
									'sentid': str(sentid),
									'lexverb': None,
									'cl.status': "Minor/Subclausal",
									'transitivity': None,
									'theme': {'text': [],
											  'headids': [],
											  'lengths': 0,
											  'functions': [],
											  'roles': [],
											  'theme_type': None, #unmarked/ marked/enhanced
											  'sub_type': None,
											  'mood_type': "Minor/Subclausal",},
									'constit': None}

	# pp.pprint(clauses, sort_dicts=False)

	return(clauses)

def extract_subtrees(head_dict: dict, doc, sent, sentid, clausehead, specialtheme, cl_status):
	'''Returns subtree (consituency) information for the clause.
	
	Parameters
	----------
	head_list : Dict
			dictionary of constituency head lists.

	Returns
	-------
			Detailed constituency info for the main verb.


	'''
	# This is the format of returned dictionary,
	# As you can see, the constituency information is extracted separately for
	# each of the sentential positions: Left or Right
	# The positional information is handy for Theme/Rheme detection, if not perfectly matching.
	constituencies = {}

	# True or False predication
	spec_theme = specialtheme['value']
	Rheme_start = specialtheme['headids']['Rheme_left']

	if cl_status == "Coordinated clause":
		conjunct = [t for t in clausehead.head.children if t.dep_ == 'cc']
		head_dict['l'] = conjunct + head_dict['l'] # Add cunjunct in the first position
		
	# First expand the input dictionary because the information is stored
	# separately for l and r. #mv is not depricated
	for position, head_list in head_dict.items():

		# Iterate over the list of constituency heads.
		for tkn in head_list:
			# Ignore any punctuations for now
			if tkn.dep_ == 'punct':
				continue

			# For any actual constituencies extract information of the subtree
			else:
				# This is to debug the difficulty in retrieving immediate phrase tag
				#print(tkn, tkn.head)

				# Issue: immediate labels cannot be retrieved easily.
				phrase = (tkn.dep_, tkn.tag_)
				root = [t for t in sent if t.dep_ == "ROOT"][0]
# =============================================================================
#				# SRL of subtree if the subhead is a verbal element.
#				if tkn.dep_ in ['ccomp', 'xcomp', 'csubj', 'advcl', 'acl', 'csubjpass', 'relcl'] and tkn.pos_ in ["VERB", 'AUX']:
#					#print(tkn.text, tkn.dep_, tkn.pos_)
#					#print([verb['tags'] for verb in doc._.srl[sentid]['verbs'] if verb['verb'] == tkn.text])
# 					try:
# 						srls = [verb['tags'] for verb in doc._.srl[sentid]['verbs'] if verb['verb'] == tkn.text][0]
# 					except IndexError:
# 						try:
# 							srls = [verb['tags'] for verb in doc._.srl[sentid]['verbs'] if verb['verb'] == tkn.head.text][0]
# 						except IndexError:
# 							try:
# 								srls = [verb['tags'] for verb in doc._.srl[sentid]['verbs'] if verb['verb'] == root.text][0]
# 							except IndexError:
# 								srls = []
# 					# print(srls)
# =============================================================================
#				else :
#					## if the head was not Verbal element, find immediate verbal element by moving up the tree one by one
#					token_copy = doc[tkn.i] # Copy the current token so that tkn variable does not change
#					while token_copy.head.pos_ not in ["VERB", 'AUX'] and token_copy.head.dep_ not in ['ROOT']:
#						token_copy = token_copy.head
#					#print(token_copy)
# =============================================================================
# 					try:
# 						srls = [verb['tags'] for verb in doc._.srl[sentid]['verbs'] if verb['verb'] == token_copy.head.text][0]
# 					except IndexError:
# 						
# 						try:
# 							srls = [verb['tags'] for verb in doc._.srl[sentid]['verbs'] if verb['verb'] == root.text][0]
# 						except IndexError:
# 							srls = []
# 				
# =============================================================================
				#defining span
				sent_start = [w for w in sent if w.is_sent_start][0] # This is necessary for srl
				# if the 
				_is_conjuncted_clause = tkn.dep_ in ["conj"] and tkn.head.dep_ in ["ROOT"] and tkn.pos_ in ["VERB", "AUX"]
				if tkn.dep_ in ["ROOT"] or _is_conjuncted_clause: #should not repeat the whole phrases
					left_ed = tkn.i
					right_ed = tkn.i + 1
				
				else:
					left_ed = tkn.left_edge.i
					right_ed = tkn.right_edge.i + 1

				# if the sentence has special theme, update the span for constituency
				if spec_theme:
					if tkn.dep_ == 'ROOT':
						left_ed = tkn.i
						right_ed = tkn.i + 1

					elif left_ed < float(Rheme_start) < right_ed:
						right_ed = Rheme_start

				#srl = srls[left_ed - sent_start.i:right_ed - sent_start.i]

				# This is another implementation of subtree extraction

				const_span = doc[left_ed:right_ed]
				# print(const_span)
				text = const_span #changed into span
				tokenid = [int(w.i) for w in const_span]
				# lemma = [w.lemma_ for w in const_span]
				pos = [str(w.pos_) for w in const_span]
				tag = [str(w.tag_) for w in const_span]
				dep = [str(w.dep_) for w in const_span]
				ent = [str(w.ent_type_) for w in const_span]
				# pos.extend([w.pos_ for w in doc[left_ed:right_ed+1]])
					# Consider adding SRL here.. -> needs to incorporate AllenNLP

				constituency_info = {"text": text,  # extract text
									 'tokenid': tokenid,
									 'start_id': int(left_ed),
									 'end_id': int(right_ed),
									 'syn_head': phrase,  # label for syntactic node
									 'relation': tkn.dep_,  # dependency relation to the head
									 "pos": pos,
									 "tag": tag,
									 "dep": dep,
									 "ent": ent,
									 #"srl_tag": srl,
									 "position": position,  # Left or right in
									 "gov": tkn.head.text,  # head as text
									 }
				
				
				constituencies[tkn.i] = constituency_info

	constituencies =  OrderedDict(sorted(constituencies.items())) #sorted(d, key=d.get)
	return(constituencies)

### constituency analysis end ##


### Theme/Rheme analysis ###
intj = ['i mean', 'you know', 'i know']
topical_ent = ("PERSON", "FAC", "ORG", "GPE", 'LOC', 'PRODUCT', 'EVENT', 'WORK_OF_ART', "LAW", "LANGUAGE", "DATE", "TIME", 'PERCENT', "MONEY", "QUANTITY")


def theme_rules(mainverb, headid, group):
	'''This is conditions for theme function and role
	
	Parameters
	----------
	mainverb : TYPE
		DESCRIPTION.
	headid : TYPE
		DESCRIPTION.
	group : TYPE
		DESCRIPTION.

	Returns
	-------
	None.

	'''
	flag_for_check = False
	function = 'Error'
	role = group['relation']
	group_text = group['text'].text
	
	if group['relation'] in ['npadvmod'] and 'PROPN' in group['pos']:  # Vocative
	
		function = 'Interpersonal'
		role = 'vocative'
	
	# Textual theme 
	## continuatives e.g., yes, no well, oh 
	## some continuatives are advmod e.g., now 
	elif any(tag in  group['dep'] for tag in ['intj']): # or 'B-ARGM-DIS' in group['srl_tag']:  # textual theme
		function = 'Textual'
		role = group['relation']
	
	
	# Textual theme 
	## conjunction  e.g., and or nor either neither but yet so 
	## some cunjunctions are advmod then.
	elif group['relation'] in ['cc'] :  # textual theme
		function = 'Textual'
		role = group['relation']
	
	#Add I mean, you know
	
	elif group_text.lower() in intj and group['relation'] in ['parataxis']:
		function = 'textual'
		role = group['relation']

	# Wh-questions 
	elif 'Wh- Interrogative' in mainverb['theme']['mood_type'] and any(tag in group['tag'] for tag in ("WDT", "WP", "WP$", "WRB")):  # WH questions
		#print("WH branch activated")
		function = 'Interpersonal/Topical'
		role = group['relation']
	
	elif mainverb['theme']['mood_type'] == "Imperative" and group['text'].text.lower() == "let":
		function = 'Interpersonal'
		role = group['relation']
	
	# disambiguating textual vs interpersonal
	elif group['relation'] in ['advmod', 'prep', 'mark']: 
		flag_for_check = True
		
		# Perfect match by list of modal and conjunctive adjuncts
		if group_text.lower() in modal_flat:
			function = 'Interpersonal'
			
		elif group_text.lower() in conjunctive_flat:
			function = 'Textual'
		
		## should insert multiword modal and conjunctive adjuncts here 
		## In the further this should also look at the grammatical context in which the mwu occur
		#elif any(mwu in group['text'].lower() for mwu in multiword_modal):
			#function = 'Interpersonal'
			
		#elif any(mwu in group['text'].lower() for mwu in multiword_conjunct):
			#function = 'Textual'
		
		elif any(tag in group['tag'] for tag in ("CD", "NNP", "NNPS", "WRB")):
			function = 'Topical'
			
		elif any(ent in group['ent'] for ent in topical_ent):
			function = 'Topical'
		
		
		else:
			function = 'Topical' ## current implementation
			
		role = group['relation']

	# Aux is interpersonal, in question particularly
	elif group['relation'] in ['aux']:
		function  = 'Interpersonal'
		role = group['relation']

	# Topical theme: Participants or circumstances
	elif group['relation'] in ['nsubj', 'nsubjpass', 'csubj', 'csubjpass', 'dobj', 'attr', 'expl', "ccomp", "xcomp", 'acomp', 'advcl', 'npadvmod', "npmod", 'agent', "dative"]:  # Topical theme
		function = "Topical"
		role = group['relation']
		
	# Dislocated participants (dep and nominal elements)
	## Make sure this is not confused with vocative
	elif group['relation'] in [ "obl","pcomp", "pobj", "obj", "pobj", 'dep', 'preconj']:  # Topical theme
		flag_for_check = True
		
		if any(ent in group['ent'] for ent in topical_ent): #entity recognition
			function = 'Topical'
		else:
			function = "Topical??"
		role = group['relation']

	# THis is for imparative...? But may not be avtivated 
	elif group['relation'] == "ROOT":
		if mainverb['theme']['mood_type'] == "Yes/No Interrogative":
			function = 'Interpersonal'
		if mainverb['theme']['mood_type'] == "Imperative":
			function = 'Topical'
		role = group['relation']

	# Negation
	elif group['relation'] == "neg":
		function = 'Interpersonal'
		role = group['relation']
	
	elif group['relation'] == "parataxis":
		function = 'Topical'
		role = group['relation']
	
	else:
		flag_for_check = True
		function = "ERROR"
		role = group['relation']
	
	return (function, role, flag_for_check)

def extract_theme(parsed_clauses, doc, print_res=True, collect_ambiguous = False):
	
	
	for id, mainverb in parsed_clauses.items():
		
		#** structure of main verb **
		#{'lexverb': 'is',
	    # 'cl.status': 'ROOT',
	    # 'theme': {'headids': [],
	    #           'functions': [],
	    #           'roles': [],
	    #           'marked': False,
	    #           'mood_type': 'Declarative'},
	    # 'srl_tags': [],
		# 'constit': {}}
		
		if mainverb['cl.status'] in ['Coordinated VP']:
			continue
		elif mainverb['lexverb'] == None:
			continue
		else:
			special_theme = mainverb['theme']['special_theme']['value']
			special_type = mainverb['theme']['special_theme']['type']
			
			
			
			## iterate sorted dictionary...?
				
			if special_theme:
				
				branch = False #when this is True, the information is stored as special theme
				special_text = []
				special_headid = []
				
				normal_text = []
				normal_headid = []
				normal_funct = []
				normal_role = []
				
				funct_holder = []
				
				for headid, group in mainverb['constit'].items():
					#token = doc[group['token_id']]
					group_text = group['text'] #this is raw text
					
					if special_type == 'predicated':
						special_funct  = 'Topical'
						
						if group['position'] == 'l':
							
							if not branch:
							# predicated part is from it subject to the end of the predicated
								if group['text'].text.lower() == "it" and group['relation'] in ['nsubj', 'nsubjpass']:
									special_text.append(group['text'])
									special_headid.extend(group['tokenid'])
									branch = True
									
								else: #this should be whatever comes before
									function, role, flag_for_check = theme_rules(mainverb, headid, group)
									normal_text.append((group['text'], headid)) #spacy span
									normal_headid.append((headid, headid))
									normal_funct.append((function, headid))
									normal_role.append((role, headid))
									mainverb['theme']['lengths'] += len(group['tokenid'])
									funct_holder.append(function)
									
									## output ambiguous theme into file
									if flag_for_check and collect_ambiguous:
										collect_ambiguous_theme(parsed_clauses, doc, mainverb, group, function, role)
									
							else:
								special_text.append(group['text'])
								special_headid.extend(group['tokenid'])
					
					if special_type == "thematized comment":
						special_funct  = 'Interpersonal'
	
						if group['position'] == 'l':
							
							if not branch:
							# predicated part is from it subject to the end of the predicated
								if group['text'].text.lower() == "it" and group['relation'] in ['nsubj', 'nsubjpass']:
									special_text.append(group['text'])
									special_headid.extend(group['tokenid'])
									
									branch = True
								else: #this should be whatever comes before and after the thematized comment
									if ('Topical' not in funct_holder and 'Interpersonal/Topical' not in funct_holder):
										function, role, flag_for_check = theme_rules(mainverb, headid, group)
										normal_text.append((group['text'], headid)) #spacy apan
										normal_headid.append((headid, headid))
										normal_funct.append((function, headid))
										normal_role.append((role, headid))
										mainverb['theme']['lengths'] += len(group['tokenid'])
										# to stop adding two Topical theme
										funct_holder.append(function)

										## output ambiguous theme into file
										
										if flag_for_check and collect_ambiguous:
											collect_ambiguous_theme(parsed_clauses, doc, mainverb, group, function, role)
							else:
								special_text.append(group['text'])
								special_headid.extend(group['tokenid'])
	
								if group['text'].text.lower() == "that" and group['relation'] in ['mark']:
									branch = False
									
					
				## Sorting here
				#combined_text = [(" ".join(special_text), safe_devide(sum(special_headid),len(special_headid)))] + normal_text
				combined_text = [(doc[special_text[0].start:special_text[-1].end], safe_devide(sum(special_headid),len(special_headid)))] + normal_text
				#combined_text = special_text + normal_text
				combined_headid = [(special_headid, safe_devide(sum(special_headid),len(special_headid)))] + normal_headid
				combined_funct = [(special_funct, safe_devide(sum(special_headid),len(special_headid)))] + normal_funct
				combined_role = [(special_type, safe_devide(sum(special_headid),len(special_headid)))] + normal_role
				
				combined_text = [w[0] for w in sorted(combined_text, key=itemgetter(1))] #this was the bug
				
				combined_headid = [str(w[0]) for w in sorted(combined_headid, key=itemgetter(1))]
				combined_funct = [str(w[0]) for w in sorted(combined_funct, key=itemgetter(1))]
				combined_role = [str(w[0]) for w in sorted(combined_role, key=itemgetter(1))]
	 			
				##update
				mainverb['theme']['text'] = combined_text
				mainverb['theme']['headids'] = combined_headid
				mainverb['theme']['functions'] = combined_funct
				mainverb['theme']['roles'] = combined_role
				mainverb['theme']['lengths'] += len(special_headid)
					
									
			else:
				mainverb['theme']['theme_type'] = 'Unmarked'
				
				for headid, group in mainverb['constit'].items():
					if ('Topical' not in mainverb['theme']['functions'] and 'Interpersonal/Topical' not in mainverb['theme']['functions']): # or special_theme:
						
						
						if group['position'] == 'l':
							
							#√ 1. Identify mood class of the clause (e.g. declarative, interrogative, or imperative)
							#√ 2. syntactic node (e.g. NP, VP, or ADVP)
							#√ 3. theme function (topical, interpersonal, or textual),
							#√ Clause-level 4. markedness (marked or unmarked),
							#√ Clause-level 5. mood type, and
							#√ 6. theme role (e.g. Subject, Complement or Adjunct)
							# 7. Syntactic complexity of theme
							
							# predicated theme has already been flagged
							
							##### pass the info onto conditional branches that return function and role ####
							function, role, flag_for_check = theme_rules(mainverb, headid, group)
											
							##### Store the information here #####
							# 1) first for the theme info
							mainverb['theme']['text'].append(group['text'])
							mainverb['theme']['headids'].append(headid)
							mainverb['theme']['functions'].append(function)
							mainverb['theme']['roles'].append(role)
							mainverb['theme']['lengths'] += len(group['tokenid'])
							
							if function == 'Topical':
								mainverb['theme']['sub_type'] = role
							
							# 2) add the info to the constituents dictionary as well
							mainverb['constit'][headid]['theme_function'] = function
							mainverb['constit'][headid]['theme_role'] = role
							
							## output ambiguous theme into file
							if flag_for_check and collect_ambiguous:
								collect_ambiguous_theme(parsed_clauses, doc, mainverb, group, function, role)
							
						# if imparative, add the main verb as the Topical theme
						elif mainverb['theme']['mood_type'] == 'Imparative':
							mainverb['theme']['text'].append(mainverb['lexverb'])
							mainverb['theme']['headids'].append(id)
							mainverb['theme']['functions'].append('Interpersonal/Topical')
							mainverb['theme']['roles'].append('ROOT')
							mainverb['theme']['lengths'] += len(group['tokenid'])
							mainverb['theme']['sub_type'] = 'ROOT'

							mainverb['constit'][headid]['theme_function'] = 'Interpersonal/Topical'
							mainverb['constit'][headid]['theme_role'] = "ROOT"
							
			
			# Detect Theme type
			# 
			# unmarked theme = Topical theme with subject
			# marked theme = Topical theme with complement or adjunct
			# enhanced theme includes predicated, equative and thematized comment
			for func, synrole in zip(mainverb['theme']['functions'], mainverb['theme']['roles']):
				if func in ['Interpersonal', 'Textual']:
					continue
				
				elif mainverb['theme']['mood_type'] not in ['Imperative']:
					
					if func in ['Topical'] and synrole not in ['nsubj', 'nsubjpass', 'csubj', 'csubjpass']:
							mainverb['theme']['theme_type'] = 'Marked'
							mainverb['theme']['sub_type'] = synrole
							
				elif mainverb['theme']['mood_type'] in ['Imperative']:
					if func in ['Topical'] and synrole not in ['ROOT', 'conj']:
							mainverb['theme']['theme_type'] = 'Marked'
							mainverb['theme']['sub_type'] = synrole
						#mainverb['theme']['marked'] = True #migrated with theme type
						#mainverb['theme']['sub_type'] = synrole
	
			# Detect Thematic equatives
			for func, synrole, headids in zip(mainverb['theme']['functions'], mainverb['theme']['roles'], mainverb['theme']['headids']):
				if func in ['Interpersonal', 'Textual']:
					continue
				elif func in ['Topical'] and synrole in ['csubj', 'csubjpass']:
					for (dep, word) in [(w.dep_, w.text.lower()) for w in doc[int(headids)].subtree]:
						if dep in ['nsubj', 'nsubjpass', 'dobj', 'iobj', 'attr', 'det'] and word == 'what':
							mainverb['theme']['theme_type'] = 'Enhanced'
							mainverb['theme']['sub_type'] = 'Equative'
							#mainverb['theme']['equative'] = True #migrated with theme type
	
			if special_type == 'predicated':
				mainverb['theme']['theme_type'] = 'Enhanced'
				mainverb['theme']['sub_type'] = 'Predicated'

			if special_type == 'thematized comment':
				mainverb['theme']['theme_type'] = 'Enhanced'
				mainverb['theme']['sub_type'] = 'Thematized comment'

			if print_res == True:
				print("Theme analysis of ", mainverb['lexverb'])
				pp.pprint(mainverb['theme'], sort_dicts=False)

	# pp.pprint(parsed_clauses, sort_dicts=False)
	return(parsed_clauses)

def extract_theme_span(annotated_tree: dict):
	holder = []
	result = {}
	for headid, tunit in annotated_tree.items():
		
		for thmid, theme in enumerate(tunit['theme']['text']):
			#st.write(theme)
			for w in theme:
				#st.write(w)
				result[w.i] = tunit['theme']['functions'][thmid]
				
			holder.extend([w.i for w in theme])
	#st.write(result)
	return (result)




def collect_ambiguous_theme(parsed_clauses, doc, mainverb, group, function, role):
	test_dir = '0_basenlp/SFLAnalyzer/test_parse/'
	outputname = test_dir + "ambiguous_themes_0713_tsv.txt"
	writeHeader = False
	if not path.exists(outputname):
		#header = 'text\tfunction\tlexverb\trole\tSRL\ttag\tdep\tent\tsentence'
		header = 'text\tfunction\tlexverb\trole\tSRL\ttag\tdep\tent\tsentence'.split('\t')
		writeHeader = True
	holder = [] 
	
	holder.append(str(group['text']))
	holder.append(function)
	holder.append(mainverb['lexverb'])
	holder.append(role)
	
	try: 
		holder.append("_".join(group['srl_tag'])) #
	except:
		holder.append(" ")
	try: 
		holder.append("_".join(group['tag'])) #
	except:
		holder.append(" ")
	try: 
		holder.append("_".join(group['dep'])) #
	except:
		holder.append(" ")
	try: 
		holder.append("_".join(group['ent'])) #
	except:
		holder.append(" ")
		
	text = mainverb['sentence']
	text = preprocess(text)
	holder.append(str(text).strip())
	
	with open(outputname, "at") as out_file:
		tsv_writer = csv.writer(out_file, delimiter='\t')
		
		if writeHeader:
			tsv_writer.writerow(header)
		tsv_writer.writerow(holder)

def test(text, conll=False, save = True):
	if conll:
		parse_print(text)
	doc = nlp(text)
	const1 = constituent_analysis(doc)
	const1 = extract_theme(const1, doc, print_res=False)
	
	theme_span = extract_theme_span(const1)
	
	if save:
		return const1
	else:
		return pp.pprint(const1, sort_dicts=False)
	
	
def annotate(doc, spans, functions = True):
	holder= []
	temp = []
	function = 'None'
	
	for token in doc:
		
		if token.i not in spans or function != spans[token.i]:
			if len(temp) > 0:
				if functions:
					if function == "Topical":
						holder.append((" ".join(temp), function, "#fea"))
					elif function == "Interpersonal":
						holder.append((" ".join(temp), function, '#faa'))
					elif function == "Textual":
						holder.append((" ".join(temp), function, "#afa"))
						
				else:
					holder.append((" ".join(temp), function, '#faa'))
					
				holder.append(" ")
				temp = []
				
			if token.i not in spans:
				holder.append(token.text + " ")
			
		if token.i in spans:
			function = spans[token.i]
			temp.append(token.text)

	return holder


def theme_markup_test(text, conll=False, save = True):
	doc = nlp(text)
	const1 = constituent_analysis(doc)
	const1 = extract_theme(const1, doc, print_res=False)
	theme_span = extract_theme_span(const1)
	
	anno_text = annotate(doc, theme_span)
	
	return anno_text
