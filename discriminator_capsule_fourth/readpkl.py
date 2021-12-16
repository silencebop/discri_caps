# coding: utf-8

# 3/3/21 3:24 PM

import pickle

pic = open(r'scores_capsule_resnet_sampled_fer_freeze.pkl', 'rb')
data = pickle.load(pic)
print(data)

pico = open(r'scores_capsule_resnet_sampled_fer_freeze_o.pkl', 'rb')
datao = pickle.load(pico)
print(datao)



