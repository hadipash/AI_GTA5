"""
Module for testing best way to store and manage training data
"""


import h5py
import numpy
import time


def test(test_img, test_key):
    # test 1: h5, no compression    < ----------------------- Winner!
    with h5py.File("data_nocomp.h5", "w") as f:
        last_time = time.time()
        f.create_dataset("img", data=test_img)
        f.create_dataset("key", data=test_key)
        print('Writing H5 with no compression took {} seconds'.format(time.time() - last_time))

    # test 2: h5, gzip default (4)
    with h5py.File("data_gzip4comp.h5", "w") as f:
        last_time = time.time()
        f.create_dataset("img", data=test_img, compression="gzip")
        f.create_dataset("key", data=test_key, compression="gzip")
        print('Writing H5 with gzip compression (4) took {} seconds'.format(time.time() - last_time))

    # test 3: h5, gzip 2
    with h5py.File("data_gzip2comp.h5", "w") as f:
        last_time = time.time()
        f.create_dataset("img", data=test_img, compression="gzip", compression_opts=2)
        f.create_dataset("key", data=test_key, compression="gzip", compression_opts=2)
        print('Writing H5 with gzip compression (2) took {} seconds'.format(time.time() - last_time))

    # test 4: h5, lzf
    with h5py.File("data_lzfcomp.h5", "w") as f:
        last_time = time.time()
        f.create_dataset("img", data=test_img, compression="lzf")
        f.create_dataset("key", data=test_key, compression="lzf")
        print('Writing H5 with lzf compression took {} seconds'.format(time.time() - last_time))

    # test 5: npy, no compression
    last_time = time.time()
    numpy.save("data_nocomp.npy", [test_img, test_key])
    print('Writing npy with no compression took {} seconds'.format(time.time() - last_time))

    # test 6: npy, compressed
    last_time = time.time()
    numpy.savez_compressed("data_comp.npz", [test_img, test_key])
    print('Writing npz with compression took {} seconds'.format(time.time() - last_time))
