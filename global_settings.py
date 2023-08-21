in_dim = 784 # the original value
# in_dim = 20 # 784
# encoder_units = [128,128,512]
encoder_units = [512, 512, 2048]

# encoder_units = [int(0.66 * in_dim), int(0.66 * in_dim), int(2.7 * in_dim)]
# decoder_units = encoder_units.copy()
# decoder_units.reverse()
# can also specify decoder units independently of encoder_units