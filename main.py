import streamlit as st
from name_generator import NameGenerator
import random
import string

st.title('Name Generator')
"""Whether you're trying to write a character
list for an epic voyage, creating a group of witches
for a children's book, or just trying to name your pet
unicorn, we'll find the name for you."""

rnn = NameGenerator()
race = st.selectbox(label="Race", options=('Man', 'Elf', 'Dwarf', 'Hobbit', 'Ainur'))
username = st.text_input(label="Starting Characters" ,value='', key=None, type='default')

if username == '':
    username = random.choice(string.ascii_letters).upper()
st.text('Generated name is : ' + rnn.generate_name(race,username))
