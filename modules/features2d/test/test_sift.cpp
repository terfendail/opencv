// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

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
