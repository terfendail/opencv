/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                           License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
// Copyright (C) 2009, Willow Garage Inc., all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of the copyright holders may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

#include "test_precomp.hpp"

namespace opencv_test { namespace {

class CV_SiftTest : public cvtest::BaseTest
{
public:
    CV_SiftTest();
    ~CV_SiftTest();
protected:
    void run(int);
};

CV_SiftTest::CV_SiftTest() {}
CV_SiftTest::~CV_SiftTest() {}

void CV_SiftTest::run( int )
{
    Mat image = imread(string(ts->get_data_path()) + "features2d/tsukuba.png");
    string xml = string(ts->get_data_path()) + "sift/result.xml.gz";

    if (image.empty())
    {
        ts->printf( cvtest::TS::LOG, "No image.\n" );
        ts->set_failed_test_info( cvtest::TS::FAIL_INVALID_TEST_DATA );
        return;
    }

    Mat gray;
    cvtColor(image, gray, COLOR_BGR2GRAY);

    vector<KeyPoint> calcKeypoints;
    Mat calcDescriptors;
    Ptr<SIFT> sift = cv::SIFT::create();
    sift->detectAndCompute(gray, Mat(), calcKeypoints, calcDescriptors, false);

    FileStorage fs(xml, FileStorage::READ);
    if (!fs.isOpened())
    {
        std::cout << "Creating xml..." << std::endl;
        fs.open(xml, FileStorage::WRITE);
        if (!fs.isOpened())
        {
            ts->set_failed_test_info(cvtest::TS::FAIL_INVALID_TEST_DATA);
            return;
        }
        fs << "keypoints" << calcKeypoints;
        fs << "descriptors" << calcDescriptors;
        fs.release();
        fs.open(xml, FileStorage::READ);
        if (!fs.isOpened())
        {
            ts->set_failed_test_info(cvtest::TS::FAIL_INVALID_TEST_DATA);
            return;
        }
    }

    vector<KeyPoint> validKeypoints;
    Mat validDescriptors;
    read( fs["keypoints"], validKeypoints );
    read( fs["descriptors"], validDescriptors, Mat() );
    fs.release();

    // Compare the number of keypoints
    if (validKeypoints.size() != calcKeypoints.size())
    {
        ts->printf( cvtest::TS::LOG, "Bad keypoints count (validCount = %d, calcCount = %d).\n",
                    validKeypoints.size(), calcKeypoints.size() );
        ts->set_failed_test_info(cvtest::TS::FAIL_MISMATCH);
        return;
    }

    // Compare the order and coordinates of keypoints
    {
        size_t exactCount = 0;
        for (size_t i = 0; i < validKeypoints.size(); i++) {
            float dist = (float)cv::norm( calcKeypoints[i].pt - validKeypoints[i].pt );
            if (dist == 0) {
                exactCount++;
            }
        }
        if (exactCount < validKeypoints.size()) {
            ts->printf( cvtest::TS::LOG, "Keypoints mismatch: exact count (dist==0) is %d/%d.\n",
                    exactCount, validKeypoints.size() );
            ts->set_failed_test_info(cvtest::TS::FAIL_MISMATCH);
            return;
        }
    }

    // Compare descriptors
    if ( validDescriptors.size != calcDescriptors.size || 0 != cvtest::norm(validDescriptors, calcDescriptors, NORM_L2))
    {
        ts->printf( cvtest::TS::LOG, "Descriptors mismatch.\n" );
        ts->set_failed_test_info(cvtest::TS::FAIL_MISMATCH);
        return;
    }

    ts->set_failed_test_info(cvtest::TS::OK);
}

TEST(Features2d_SIFT, regression_exact) { CV_SiftTest test; test.safe_run(); }

}} // namespace
