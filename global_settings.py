# -*- coding: utf-8 -*-
"""
Created on Mon Aug 14 16:10:26 2023

@author: pfbur
"""
in_dim = 784 # the original value
# in_dim = 20 # 784
# encoder_units = [128,128,512]
encoder_units = [512, 512, 2048]
decoder_units = encoder_units.copy()
decoder_units.reverse()
# can also specify decoder units independently of encoder_units