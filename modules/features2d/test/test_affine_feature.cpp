// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

#include "test_precomp.hpp"

namespace opencv_test { namespace {

bool isSimilarKeypoints( const KeyPoint& p1, const KeyPoint& p2 )
{
    const float maxPtDif = 1.f;
    const float maxSizeDif = 1.f;
    const float maxAngleDif = 2.f;
    const float maxResponseDif = 0.1f;

    float dist = (float)cv::norm( p1.pt - p2.pt );
    return (dist < maxPtDif &&
            fabs(p1.size - p2.size) < maxSizeDif &&
            abs(p1.angle - p2.angle) < maxAngleDif &&
            abs(p1.response - p2.response) < maxResponseDif &&
            p1.octave == p2.octave &&
            p1.class_id == p2.class_id );
}

TEST(Features2d_AFFINE_FEATURE, regression_py)
{
    const float badCountsRatio = 0.01f;
    const float maxBadPointsRatio = 0.03f;
    const float badDescriptorDist = 1.0f;
    const float maxBadDescriptorRatio = 0.05f;

    Mat image = imread(cvtest::findDataFile("features2d/tsukuba.png"));
    ASSERT_FALSE(image.empty());

    Mat gray;
    cvtColor(image, gray, COLOR_BGR2GRAY);
    Ptr<Feature2D> ext = AffineFeature::create(SIFT::create());
    vector<KeyPoint> calcKeypoints;
    Mat calcDescriptors;
    ext->detectAndCompute(gray, Mat(), calcKeypoints, calcDescriptors, false);

    Mat mpt, msize, mangle, mresponse, moctave;
    vector<float> pt, size, angle, response;
    vector<int> octave;
    vector<KeyPoint> validKeypoints;
    Mat validDescriptors;
    {
        // read keypoints
        string xml = cvtest::findDataFile("asift/keypoints.xml");

        FileStorage fs(xml, FileStorage::READ);
        ASSERT_TRUE(fs.isOpened()) << xml;

        fs["keypoints_pt"] >> mpt;
        fs["keypoints_size"] >> msize;
        fs["keypoints_angle"] >> mangle;
        fs["keypoints_response"] >> mresponse;
        fs["keypoints_octave"] >> moctave;
        fs.release();

        mpt.reshape(0, 1).copyTo(pt);
        msize.reshape(0, 1).copyTo(size);
        mangle.reshape(0, 1).copyTo(angle);
        mresponse.reshape(0, 1).copyTo(response);
        moctave.reshape(0, 1).copyTo(octave);

        validKeypoints.resize(pt.size());
        for( size_t i = 0; i < pt.size(); i++ )
        {
            validKeypoints[i].pt.x = pt.at(i*2);
            validKeypoints[i].pt.y = pt.at(i*2 + 1);
            validKeypoints[i].size = size.at(i);
            validKeypoints[i].angle = angle.at(i);
            validKeypoints[i].response = response.at(i);
            validKeypoints[i].octave = octave.at(i);
        }
    }
    {
        // read descriptors
        string xml = cvtest::findDataFile("asift/descriptors.xml");

        FileStorage fs(xml, FileStorage::READ);
        ASSERT_TRUE(fs.isOpened()) << xml;

        fs["descriptors"] >> validDescriptors;
        fs.release();
    }

    // compare keypoints
    float countRatio = (float)validKeypoints.size() / (float)calcKeypoints.size();
    ASSERT_LT(countRatio, 1 + badCountsRatio) << "Bad keypoints count ratio.";
    ASSERT_GT(countRatio, 1 - badCountsRatio) << "Bad keypoints count ratio.";

    int badPointCount = 0;
    vector<size_t> correspond(validKeypoints.size());
    for( size_t v = 0; v < validKeypoints.size(); v++ )
    {
        int nearestIdx = -1;
        float minDist = std::numeric_limits<float>::max();

        for( size_t c = 0; c < calcKeypoints.size(); v++ )
        {
            float curDist = (float)cv::norm( calcKeypoints[c].pt - validKeypoints[v].pt );
            if( curDist < minDist )
            {
                minDist = curDist;
                nearestIdx = (int)c;
            }
        }

        correspond[v] = nearestIdx;
        if( !isSimilarKeypoints( validKeypoints[v], calcKeypoints[nearestIdx] ) )
            badPointCount++;
    }
    float badPointRatio = (float)badPointCount / (float)validKeypoints.size();
    ASSERT_LT( badPointRatio, maxBadPointsRatio ) << "Too many keypoints mismatched.";

    // Compare descriptors
    int dim = validDescriptors.cols;
    CV_Assert( CV_32F == validDescriptors.type() );
    int badDescriptorCount = 0;
    L1<float> distance;
    for( size_t v = 0; v < validKeypoints.size(); v++ )
    {
        size_t c = correspond[v];
        float dist = distance( validDescriptors.ptr<float>(v), calcDescriptors.ptr<float>(c), dim );
        if( dist > badDescriptorDist )
            badDescriptorCount++;
    }
    float badDescriptorRatio = (float)badDescriptorCount / (float)validKeypoints.size();
    ASSERT_LT( badDescriptorRatio, maxBadDescriptorRatio ) << "Too many descriptors mismatched.";
}

}} // namespace
