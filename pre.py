"""Labeling"""

import cv2
import glob
import sys
import os


def main(dirw):
    dir_name = dirw[1]
    CT_PATH = './' + dir_name + '/CT'
    MR_PATH = './' + dir_name + '/MR'
    CT_LABEL_PATH = './' + dir_name + '/CT_MASK'
    MR_LABEL_PATH = './' + dir_name + '/MR_MASK'

    NEW_CT = './' + dir_name + '/CT_INTENSITY'
    NEW_MR = './' + dir_name + '/MR_INTENSITY'
    NEW_CTL = './' + dir_name + '/CT_LABEL'
    NEW_MRL = './' + dir_name + '/MR_LABEL'

    CT_files = glob.glob(CT_PATH + '/*.png')
    CT_files.sort()
    MR_files = glob.glob(MR_PATH + '/*.png')
    MR_files.sort()
    CTMS_files = glob.glob(CT_LABEL_PATH + '/*.png')
    CTMS_files.sort()
    MRMS_files = glob.glob(MR_LABEL_PATH + '/*.png')
    MRMS_files.sort()

    print(CT_files)
    print(MR_files)
    print(CTMS_files)
    print(MRMS_files)

    for image_num in range(len(CTMS_files)):
        ct_label = cv2.imread(CTMS_files[image_num])
        mr_label = cv2.imread(MRMS_files[image_num])
        new_ct_label = cv2.cvtColor(cv2.imread(CTMS_files[image_num]),
                                    cv2.COLOR_BGR2GRAY)
        new_mr_label = cv2.cvtColor(cv2.imread(MRMS_files[image_num]),
                                    cv2.COLOR_BGR2GRAY)
        print('Generate GRAY')
        height, width, chanel = ct_label.shape
        for yy in range(height):
            for xx in range(width):
                pix = mr_label[yy, xx]
                if all(pix == [127, 0, 127]):
                    # purple(CLASS 1)
                    new_mr_label[yy, xx] = 0
                elif all(pix == [255, 0, 255]):
                    # magenta(CLASS 2)
                    new_mr_label[yy, xx] = 63
                elif all(pix == [255, 255, 0]):
                    # syan(CLASS 3)
                    new_mr_label[yy, xx] = 127
                elif all(pix == [127, 0, 0]):
                    # dark blue(CLASS 4)
                    new_mr_label[yy, xx] = 191
                else:
                    # white(CLASS 0)
                    new_mr_label[yy, xx] = 255
                pix = ct_label[yy, xx]
                if all(pix == [0, 0, 255]):
                    # red
                    new_ct_label[yy, xx] = 0
                elif all(pix == [0, 0, 127]):
                    # brown
                    new_ct_label[yy, xx] = 64
                elif all(pix == [255, 0, 0]):
                    # blue
                    new_ct_label[yy, xx] = 128
                elif all(pix == [0, 255, 255]):
                    # yellow
                    new_ct_label[yy, xx] = 191
                else:
                    # white
                    new_ct_label[yy, xx] = 255

        ct_name = os.path.join(NEW_CTL, os.path.basename(CTMS_files[image_num]))
        cv2.imwrite(ct_name, new_ct_label)
        mr_name = os.path.join(NEW_MRL, os.path.basename(MRMS_files[image_num]))
        cv2.imwrite(mr_name, new_mr_label)
        CT_gray = cv2.cvtColor(cv2.imread(CT_files[image_num]),
                               cv2.COLOR_BGR2GRAY)
        ct_name = os.path.join(NEW_CT, os.path.basename(CT_files[image_num]))
        cv2.imwrite(ct_name, CT_gray)
        MR_gray = cv2.cvtColor(cv2.imread(MR_files[image_num]),
                               cv2.COLOR_BGR2GRAY)
        mr_name = os.path.join(NEW_MR, os.path.basename(MR_files[image_num]))
        cv2.imwrite(mr_name, MR_gray)

        print('processed ',  image_num,  '/', len(CT_files))


args = sys.argv

main(args)
