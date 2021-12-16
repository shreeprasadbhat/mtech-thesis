import os
import tensorflow

# -------------------data------------------------
data = []
with open('../data/Toy/toy.csv','r') as file :
    text = file.read()
    for line in text.splitlines() :
        data.append(line.split(','))
# print(data)
'''
# -------------------model-----------------------
class GAN(tf.keras.Model) :
    def __init__(self):
        
    def run(self):
'''
