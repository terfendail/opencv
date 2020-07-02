#!/usr/bin/env python

'''
A test for affine invariant feature-based image matching.

USAGE
  test_asift.py
'''
import os
from multiprocessing.pool import ThreadPool
import cv2 as cv
import numpy as np
from asift import affine_detect

def main():
    path = os.environ.get('OPENCV_TEST_DATA_PATH')
    fn = os.path.join(path, 'cv/features2d/tsukuba.png')
    img = cv.imread(fn, cv.IMREAD_GRAYSCALE)
    detector = cv.SIFT_create()
    pool = ThreadPool(processes = cv.getNumberOfCPUs())
    kp, desc = affine_detect(detector, img, pool=pool)

    params = [(1.0, 0.0)]
    for t in 2**(0.5*np.arange(1,6)):
        for phi in np.arange(0, 180, 72.0 / t):
            params.append((t, phi))
    params = np.array(params)

    pt = np.array(list(map(lambda p: p.pt, kp)))
    size = np.array(list(map(lambda p: p.size, kp)))
    angle = np.array(list(map(lambda p: p.angle, kp)))
    response = np.array(list(map(lambda p: p.response, kp)))
    octave = np.array(list(map(lambda p: p.octave, kp)))

    fs = cv.FileStorage(os.path.join(path, 'cv/asift/keypoints.xml'), cv.FILE_STORAGE_WRITE)
    fs.write(name='keypoints_pt', val=pt)
    fs.write(name='keypoints_size', val=size)
    fs.write(name='keypoints_angle', val=angle)
    fs.write(name='keypoints_response', val=response)
    fs.write(name='keypoints_octave', val=octave)
    fs.write(name='params', val=params)
    fs.release()

    fs = cv.FileStorage(os.path.join(path, 'cv/asift/descriptors.xml'), cv.FILE_STORAGE_WRITE)
    fs.write(name='descriptors', val=desc)
    fs.release()

if __name__ == '__main__':
    print(__doc__)
    main()
