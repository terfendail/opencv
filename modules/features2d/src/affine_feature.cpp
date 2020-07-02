/*********************************************************************
* Software License Agreement (BSD License)
*
*  Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
*  Copyright (C) 2008-2013, Willow Garage Inc., all rights reserved.
*  Copyright (C) 2013, Evgeny Toropov, all rights reserved.
*  Third party copyrights are property of their respective owners.
*
*  Redistribution and use in source and binary forms, with or without
*  modification, are permitted provided that the following conditions
*  are met:
*
*   * Redistributions of source code must retain the above copyright
*     notice, this list of conditions and the following disclaimer.
*   * Redistributions in binary form must reproduce the above
*     copyright notice, this list of conditions and the following
*     disclaimer in the documentation and/or other materials provided
*     with the distribution.
*   * The name of the copyright holders may not be used to endorse
*     or promote products derived from this software without specific
*     prior written permission.
*
*  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
*  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
*  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
*  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
*  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
*  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
*  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
*  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
*  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
*  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
*  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
*  POSSIBILITY OF SUCH DAMAGE.
*********************************************************************/

/*
 Guoshen Yu, Jean-Michel Morel, ASIFT: An Algorithm for Fully Affine
 Invariant Comparison,  Image Processing On Line, 1 (2011), pp. 11–38.
 https://doi.org/10.5201/ipol.2011.my-asift
 */

#include "precomp.hpp"
#include <iostream>
namespace cv {

class AffineFeature_Impl CV_FINAL : public AffineFeature
{
public:
    explicit AffineFeature_Impl(const Ptr<Feature2D>& backend,
            int maxTilt, int minTilt);

    int descriptorType() const CV_OVERRIDE
    {
        return backend_->descriptorType();
    }

    int defaultNorm() const CV_OVERRIDE
    {
        return backend_->defaultNorm();
    }

    void detectAndCompute(InputArray image, InputArray mask, std::vector<KeyPoint>& keypoints,
            OutputArray descriptors, bool useProvidedKeypoints=false) CV_OVERRIDE;

    void setViewParams(const std::vector<float>& tilts, const std::vector<float>& rolls) CV_OVERRIDE;
    void getViewParams(std::vector<float>& tilts, std::vector<float>& rolls) const CV_OVERRIDE;

protected:
    void affineSkew(float tilt, float roll,
            const Mat& image, const Mat& mask,
            Mat& warpedImage, Mat& warpedMask,
            Matx23f& pose) const;

    void splitKeypointsByView(const std::vector<KeyPoint>& keypoints_,
            std::vector< std::vector<KeyPoint> >& keypointsByView) const;

    const Ptr<Feature2D> backend_;
    int maxTilt_;
    int minTilt_;

    // Tilt sampling step \delta_t in Algorithm 1 in the paper.
    float tiltStep_ = 1.4142135623730951f;
    // Rotation sampling step factor b in Algorithm 1 in the paper.
    float rotateStepBase_ = 72;

    // Tilt factors.
    std::vector<float> tilts_;
    // Roll factors.
    std::vector<float> rolls_;

private:
    AffineFeature_Impl(const AffineFeature_Impl &); // copy disabled
    AffineFeature_Impl& operator=(const AffineFeature_Impl &); // assign disabled
};

AffineFeature_Impl::AffineFeature_Impl(const Ptr<FeatureDetector>& backend,
        int maxTilt, int minTilt)
    : backend_(backend), maxTilt_(maxTilt), minTilt_(minTilt)
{
    int i = minTilt_;
    if( i == 0 )
    {
        tilts_.push_back(1);
        rolls_.push_back(0);
        i++;
    }
    double tilt = 1;
    for( ; i <= maxTilt_; i++ )
    {
        tilt *= tiltStep_;
        float rotateStep = rotateStepBase_ / tilt;
        int rollN = cvFloor(180.0f / rotateStep);
        if( rollN * rotateStep == 180.0f )
            rollN--;
        for( int j = 0; j <= rollN; j++ )
        {
            tilts_.push_back(tilt);
            rolls_.push_back(rotateStep * j);
        }
    }
}

void AffineFeature_Impl::setViewParams(const std::vector<float>& tilts,
        const std::vector<float>& rolls)
{
    CV_Assert(tilts.size() == rolls.size());
    tilts_ = tilts;
    rolls_ = rolls;
}

void AffineFeature_Impl::getViewParams(std::vector<float>& tilts,
        std::vector<float>& rolls) const
{
    tilts = tilts_;
    rolls = rolls_;
}

void AffineFeature_Impl::affineSkew(float tilt, float phi,
        const Mat& image, const Mat& mask,
        Mat& warpedImage, Mat& warpedMask, Matx23f& pose) const
{
    int h = image.size().height;
    int w = image.size().width;
    Mat rotImage;

    Mat mask0;
    if( mask.empty() )
        mask0 = Mat::ones(h, w, CV_8UC1)*255;
    else
        mask0 = mask;
    pose = Matx23f(1,0,0,
                   0,1,0);

    if( phi == 0 )
        image.copyTo(rotImage);
    else
    {
        phi = phi * CV_PI / 180;
        float s = std::sin(phi);
        float c = std::cos(phi);
        Matx22f A(c, -s, s, c);
        Matx<float, 4, 2> corners(0,0, w,0, w,h, 0,h);
        Mat tf(corners * A.t());
        Mat tcorners;
        tf.convertTo(tcorners, CV_32S);
        Rect rect = boundingRect(tcorners);
        h = rect.height; w = rect.width;
        pose = Matx23f(c, -s, -rect.x,
                       s,  c, -rect.y);
        warpAffine(image, rotImage, pose, Size(w, h), INTER_LINEAR, BORDER_REPLICATE);
    }
    if( tilt == 1 )
        warpedImage = rotImage;
    else
    {
        float s = 0.8 * sqrt(tilt * tilt - 1);
        GaussianBlur(rotImage, rotImage, Size(0, 0), s, 0.01);
        resize(rotImage, warpedImage, Size(0, 0), 1.0/tilt, 1.0, INTER_NEAREST);
        pose(0, 0) /= tilt;
        pose(0, 1) /= tilt;
        pose(0, 2) /= tilt;
    }
    if( phi != 0 || tilt != 1 )
        warpAffine(mask0, warpedMask, pose, warpedImage.size(), INTER_NEAREST);
}

void AffineFeature_Impl::splitKeypointsByView(const std::vector<KeyPoint>& keypoints_,
        std::vector< std::vector<KeyPoint> >& keypointsByView) const
{
    for( size_t i = 0; i < keypoints_.size(); i++ )
    {
        const KeyPoint& kp = keypoints_[i];
        CV_Assert( kp.class_id >= 0 && kp.class_id < (int)tilts_.size() );
        keypointsByView[kp.class_id].push_back(kp);
    }
}

void AffineFeature_Impl::detectAndCompute(InputArray _image, InputArray _mask,
        std::vector<KeyPoint>& keypoints,
        OutputArray _descriptors,
        bool useProvidedKeypoints)
{
    CV_TRACE_FUNCTION();

    bool do_keypoints = !useProvidedKeypoints;
    bool do_descriptors = _descriptors.needed();
    Mat image = _image.getMat(), mask = _mask.getMat();
    Mat descriptors;

    if( (!do_keypoints && !do_descriptors) || _image.empty() )
        return;

    std::vector< std::vector<KeyPoint> > keypointsCollection(tilts_.size());
    std::vector< Mat > descriptorCollection(tilts_.size());

    if( do_keypoints )
        keypoints.clear();
    else
        splitKeypointsByView(keypoints, keypointsCollection);

    for( size_t i = 0; i < tilts_.size(); i++ )
    {
        Mat warpedImage, warpedMask;
        Matx23f pose, invPose;
        affineSkew(tilts_[i], rolls_[i], image, mask, warpedImage, warpedMask, pose);
        invertAffineTransform(pose, invPose);

        std::vector<KeyPoint> wKeypoints;
        Mat wDescriptors;
        if( !do_keypoints )
        {
            const std::vector<KeyPoint>& keypointsInView = keypointsCollection[i];
            std::vector<Point2f> pts_, pts;
            KeyPoint::convert(keypointsInView, pts_);
            transform(pts_, pts, pose);
            wKeypoints.resize(keypointsInView.size());
            for( size_t wi = 0; wi < wKeypoints.size(); wi++ )
            {
                wKeypoints[wi] = keypointsInView[wi];
                wKeypoints[wi].pt = pts[wi];
            }
        }
        backend_->detectAndCompute(warpedImage, warpedMask, wKeypoints, wDescriptors, useProvidedKeypoints);
        if( do_keypoints )
        {
            // KeyPointsFilter::runByPixelsMask( wKeypoints, warpedMask );
            std::vector<Point2f> pts_, pts;
            KeyPoint::convert(wKeypoints, pts_);
            transform(pts_, pts, invPose);

            keypointsCollection[i].resize(wKeypoints.size());
            for( size_t wi = 0; wi < wKeypoints.size(); wi++ )
            {
                keypointsCollection[i][wi] = wKeypoints[wi];
                keypointsCollection[i][wi].pt = pts[wi];
                keypointsCollection[i][wi].class_id = i;
            }
        }
        if( do_descriptors )
            wDescriptors.copyTo(descriptorCollection[i]);
    }

    if( do_keypoints )
        for( auto& keys : keypointsCollection )
            keypoints.insert(keypoints.end(), keys.begin(), keys.end());

    if( do_descriptors )
    {
        _descriptors.create(keypoints.size(), backend_->descriptorSize(), backend_->descriptorType());
        descriptors = _descriptors.getMat();
        int iter = 0;
        for( auto& descs : descriptorCollection )
        {
            Mat roi(descriptors, Rect(0, iter, descriptors.cols, descs.rows));
            descs.copyTo(roi);
            iter += descs.rows;
        }
    }
}


Ptr<AffineFeature> AffineFeature::create(const Ptr<Feature2D>& backend,
                                         int maxTilt, int minTilt)
{
    CV_Assert(minTilt < maxTilt);
    return makePtr<AffineFeature_Impl>(backend, maxTilt, minTilt);
}

String AffineFeature::getDefaultName() const
{
    return (Feature2D::getDefaultName() + ".AffineFeature");
}

} // namespace
