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
            (p1.octave & 0xffff) == (p2.octave & 0xffff)     // do not care about sublayers, class_id
            );
}

TEST(Features2d_AFFINE_FEATURE, regression_py)
{
    const float badCountsRatio = 0.01f;
    const float badDescriptorDist = 1.0f;
    const float maxBadDescriptorRatio = 0.05f;

    Mat image = imread(cvtest::findDataFile("features2d/tsukuba.png"));
    ASSERT_FALSE(image.empty());

    Mat gray;
    cvtColor(image, gray, COLOR_BGR2GRAY);

    Mat mpt, msize, mangle, mresponse, moctave, mparams;
    vector<float> tilts, rolls;
    vector<KeyPoint> validKeypoints;
    Mat validDescriptors;
    {
        // read keypoints
        string xml = cvtest::findDataFile("asift/keypoints.xml");

        FileStorage fs(xml, FileStorage::READ);
        ASSERT_TRUE(fs.isOpened()) << xml;

        fs["keypoints_pt"] >> mpt;
        ASSERT_EQ(mpt.type(), CV_64F);
        fs["keypoints_size"] >> msize;
        ASSERT_EQ(msize.type(), CV_64F);
        fs["keypoints_angle"] >> mangle;
        ASSERT_EQ(mangle.type(), CV_64F);
        fs["keypoints_response"] >> mresponse;
        ASSERT_EQ(mresponse.type(), CV_64F);
        fs["keypoints_octave"] >> moctave;
        ASSERT_EQ(moctave.type(), CV_32S);
        fs["params"] >> mparams;
        ASSERT_EQ(mparams.type(), CV_64F);

        validKeypoints.resize(mpt.rows);
        std::cout << msize.rows << "x" << msize.cols << std::endl;
        for( size_t i = 0; i < validKeypoints.size(); i++ )
        {
            validKeypoints[i].pt.x = mpt.at<double>(i,0);
            validKeypoints[i].pt.y = mpt.at<double>(i,1);
            validKeypoints[i].size = msize.at<double>(i,0);
            validKeypoints[i].angle = mangle.at<double>(i,0);
            validKeypoints[i].response = mresponse.at<double>(i,0);
            validKeypoints[i].octave = moctave.at<int>(i,0);
        }
        for( int i = 0; i < mparams.rows; i++ )
        {
            tilts.push_back(mparams.at<double>(i, 0));
            rolls.push_back(mparams.at<double>(i, 1));
        }
        fs.release();
    }
    {
        // read descriptors
        string xml = cvtest::findDataFile("asift/descriptors.xml");

        FileStorage fs(xml, FileStorage::READ);
        ASSERT_TRUE(fs.isOpened()) << xml;

        fs["descriptors"] >> validDescriptors;
        fs.release();
    }

    // calculate
    Ptr<AffineFeature> ext = AffineFeature::create(SIFT::create());
    ext->setViewParams(tilts, rolls);
    vector<KeyPoint> calcKeypoints;
    Mat calcDescriptors;
    ext->detectAndCompute(gray, Mat(), calcKeypoints, calcDescriptors, false);

    // compare keypoints
    std::cout << validKeypoints.size() << " " << calcKeypoints.size() << std::endl;
    float countRatio = (float)validKeypoints.size() / (float)calcKeypoints.size();
    ASSERT_LT(countRatio, 1 + badCountsRatio) << "Bad keypoints count ratio.";
    ASSERT_GT(countRatio, 1 - badCountsRatio) << "Bad keypoints count ratio.";

    int badPointCount = 0, commonPointCount = max((int)validKeypoints.size(), (int)calcKeypoints.size());
    vector<size_t> correspond(validKeypoints.size());
    for( size_t v = 0; v < validKeypoints.size(); v++ )
    {
        int nearestIdx = -1;
        float minDist = std::numeric_limits<float>::max();

        for( size_t c = 0; c < calcKeypoints.size(); c++ )
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
    ASSERT_LT( badPointCount, 0.9 * commonPointCount ) << "Bad accuracy!";

    // Compare descriptors
    int dim = validDescriptors.cols;
    ASSERT_EQ( CV_32F, validDescriptors.type() );
    std::cout << validDescriptors.size() << std::endl;
    ASSERT_EQ( CV_32F, calcDescriptors.type() );
    std::cout << calcDescriptors.size() << std::endl;
    int badDescriptorCount = 0;
    L1<float> distance;
    for(int i=0; i<dim; i++) std::cout << validDescriptors.ptr<float>(0)[i] <<" ";
    std::cout << std::endl;
    for(int i=0; i<dim; i++) std::cout << calcDescriptors.ptr<float>(0)[i] << " ";
    std::cout << std::endl;

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
