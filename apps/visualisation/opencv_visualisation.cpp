#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/optflow.hpp>

#include <iostream>
#include <stdio.h>

using namespace cv;
using namespace std;
//using namespace optflow;

namespace {

Mat convertFlow(const Mat & flow)
{
    Mat flow_split[2];
    Mat magnitude, angle;
    Mat hsv_split[3], hsv, rgb;
    split(flow, flow_split);
    cartToPolar(flow_split[0], flow_split[1], magnitude, angle, true);
    normalize(magnitude, magnitude, 0, 1, NORM_MINMAX);
    hsv_split[0] = angle; // already in degrees - no normalization needed
    hsv_split[1] = Mat::ones(angle.size(), angle.type());
    hsv_split[2] = magnitude;
    merge(hsv_split, 3, hsv);
    cvtColor(hsv, rgb, COLOR_HSV2BGR);
    return rgb;
}
void flowShow(const Mat &flow, const Mat &refflow, const Mat &mask, int waitTime)
{
    printf("---> ready\n"); 
    if (waitTime)
    {
        imshow("flow", convertFlow(flow));
        imshow("flow_masked", convertFlow(flow.clone().setTo(Scalar(0, 0), 255 - mask)));
        imshow("ref", convertFlow(refflow));
        waitKey(waitTime);
    }
}

bool getSample(int im_id, Mat& img1, Mat& img2, Mat& flwn, Mat& flwo, Mat& flwn_mask, Mat& flwo_mask)
{
    // get source imgs and ref flow
    String kitty_loc("d:\\KittyFlow");
    String separator("\\");
    String img1_name = format("%s%simage_2%s%06d_10.png", kitty_loc.c_str(), separator.c_str(), separator.c_str(), im_id);
    String img2_name = format("%s%simage_2%s%06d_11.png", kitty_loc.c_str(), separator.c_str(), separator.c_str(), im_id);
    String flwn_name = format("%s%sflow_noc%s%06d_10.png", kitty_loc.c_str(), separator.c_str(), separator.c_str(), im_id);
    String flwo_name = format("%s%sflow_occ%s%06d_10.png", kitty_loc.c_str(), separator.c_str(), separator.c_str(), im_id);
    img1 = imread(img1_name, IMREAD_UNCHANGED);
    img2 = imread(img2_name, IMREAD_UNCHANGED);
    Mat sflwn = imread(flwn_name, IMREAD_UNCHANGED);
    Mat sflwo = imread(flwo_name, IMREAD_UNCHANGED);
    if (img1.empty() || img2.empty() || sflwn.empty() || sflwo.empty() ||
        img1.size() != img2.size() || img1.size() != sflwn.size() || img1.size() != sflwo.size() ||
        img1.channels() != img2.channels() || img1.depth() != CV_8U || img2.depth() != CV_8U ||
        sflwn.type() != CV_16UC3 || sflwo.type() != CV_16UC3)
        return false;
    Mat spltn_c[3], splto_c[3];
    split(sflwn, spltn_c);
    split(sflwo, splto_c);
    Mat spltn_c0 = spltn_c[0];
    Mat spltn_c1 = spltn_c[1];
    Mat spltn_c2 = spltn_c[2];
    Mat flwn_c[2], flwo_c[2];
    spltn_c[2].convertTo(flwn_c[0], CV_32F, 1. / 64, -512);
    spltn_c[1].convertTo(flwn_c[1], CV_32F, 1. / 64, -512);
    spltn_c[0].convertTo(flwn_mask, CV_8U, 255);
    splto_c[2].convertTo(flwo_c[0], CV_32F, 1. / 64, -512);
    splto_c[1].convertTo(flwo_c[1], CV_32F, 1. / 64, -512);
    splto_c[0].convertTo(flwo_mask, CV_8U, 255);
    merge(flwn_c, 2, flwn);
    merge(flwo_c, 2, flwo);
    printf("Loaded %d pair\n", im_id);
    return true;
}



vector<double> getFlow(int method, Mat img1, Mat img2, Mat& flw)
{
    int64 times[4];
    flw = Mat(img1.size(), CV_32FC2);
    switch (method)
    {
    case 0://simpleflow
        printf("Check simpleflow\n");
        {
            times[0] = getTickCount();
            //if (img1.channels() == 1)
            //{
            //    cvtColor(img1, img1, COLOR_GRAY2BGR);
            //    cvtColor(img2, img2, COLOR_GRAY2BGR);
            //}
            times[1] = getTickCount();
            Ptr<DenseOpticalFlow> algorithm = optflow::createOptFlow_SimpleFlow();
            times[2] = getTickCount();
            algorithm->calc(img1, img2, flw);
            times[3] = getTickCount();
        }
        break;
    case 1://farneback
        printf("Check farneback\n");
        {
            times[0] = getTickCount();
            if (img1.channels() == 3)
            {
                cvtColor(img1, img1, COLOR_BGR2GRAY);
                cvtColor(img2, img2, COLOR_BGR2GRAY);
            }
            times[1] = getTickCount();
            Ptr<DenseOpticalFlow> algorithm = FarnebackOpticalFlow::create();
            times[2] = getTickCount();
            algorithm->calc(img1, img2, flw);
            times[3] = getTickCount();
        }
        break;
    case 2://tvl1
        printf("Check tvl1\n");
        {
            times[0] = getTickCount();
            if (img1.channels() == 3)
            {
                cvtColor(img1, img1, COLOR_BGR2GRAY);
                cvtColor(img2, img2, COLOR_BGR2GRAY);
            }
            times[1] = getTickCount();
            Ptr<DenseOpticalFlow> algorithm = createOptFlow_DualTVL1();
            times[2] = getTickCount();
            algorithm->calc(img1, img2, flw);
            times[3] = getTickCount();
        }
        break;
    case 3://deepflow
        printf("Check deepflow\n");
        {
            times[0] = getTickCount();
            if (img1.channels() == 3)
            {
                cvtColor(img1, img1, COLOR_BGR2GRAY);
                cvtColor(img2, img2, COLOR_BGR2GRAY);
            }
            times[1] = getTickCount();
            Ptr<DenseOpticalFlow> algorithm = optflow::createOptFlow_DeepFlow();
            times[2] = getTickCount();
            algorithm->calc(img1, img2, flw);
            times[3] = getTickCount();
        }
        break;
    case 4://DISflow_ultrafast
        printf("Check DISflow_ultrafast\n");
        {
            times[0] = getTickCount();
            if (img1.channels() == 3)
            {
                cvtColor(img1, img1, COLOR_BGR2GRAY);
                cvtColor(img2, img2, COLOR_BGR2GRAY);
            }
            times[1] = getTickCount();
            Ptr<DenseOpticalFlow> algorithm = optflow::createOptFlow_DIS(optflow::DISOpticalFlow::PRESET_ULTRAFAST);
            times[2] = getTickCount();
            algorithm->calc(img1, img2, flw);
            times[3] = getTickCount();
        }
        break;
    case 5://DISflow_fast
        printf("Check DISflow_fast\n");
        {
            times[0] = getTickCount();
            if (img1.channels() == 3)
            {
                cvtColor(img1, img1, COLOR_BGR2GRAY);
                cvtColor(img2, img2, COLOR_BGR2GRAY);
            }
            times[1] = getTickCount();
            Ptr<DenseOpticalFlow> algorithm = optflow::createOptFlow_DIS(optflow::DISOpticalFlow::PRESET_FAST);
            times[2] = getTickCount();
            algorithm->calc(img1, img2, flw);
            times[3] = getTickCount();
        }
        break;
    case 6://DISflow_medium
        printf("Check DISflow_medium\n");
        {
            times[0] = getTickCount();
            if (img1.channels() == 3)
            {
                cvtColor(img1, img1, COLOR_BGR2GRAY);
                cvtColor(img2, img2, COLOR_BGR2GRAY);
            }
            times[1] = getTickCount();
            Ptr<DenseOpticalFlow> algorithm = optflow::createOptFlow_DIS(optflow::DISOpticalFlow::PRESET_MEDIUM);
            times[2] = getTickCount();
            algorithm->calc(img1, img2, flw);
            times[3] = getTickCount();
        }
        break;
    case 7://sparsetodenseflow
        printf("Check sparsetodenseflow\n");
        {
            times[0] = times[1] = getTickCount();
            Ptr<DenseOpticalFlow> algorithm = optflow::createOptFlow_SparseToDense();
            times[2] = getTickCount();
            algorithm->calc(img1, img2, flw);
            times[3] = getTickCount();
        }
        break;
    case 8://pcaflow
        printf("Check pcaflow\n");
        {
            times[0] = times[1] = getTickCount();
            Ptr<DenseOpticalFlow> algorithm = optflow::createOptFlow_PCAFlow();
            times[2] = getTickCount();
            algorithm->calc(img1, img2, flw);
            times[3] = getTickCount();
        }
        break;

    case 9://variationalRefinement
        printf("Check variationalRefinement\n");
        {
            times[0] = getTickCount();
            if (img1.channels() == 3)
            {
                cvtColor(img1, img1, COLOR_BGR2GRAY);
                cvtColor(img2, img2, COLOR_BGR2GRAY);
            }
            times[1] = getTickCount();
            Ptr<DenseOpticalFlow> algorithm = optflow::createVariationalFlowRefinement();
            times[2] = getTickCount();
            algorithm->calc(img1, img2, flw);
            times[3] = getTickCount();
        }
        break;

    case 10://LKPyrSparse
        printf("Check LKPyrSparse\n");
        {
            times[0] = getTickCount();
            vector<Point2f> prevPts, nextPts;
            vector<unsigned char> status;
            prevPts.reserve(img1.total());
            for (int j = 0; j < img1.rows; j++)
                for (int i = 0; i < img1.cols; i++)
                    prevPts.push_back(Point(i, j));
            times[1] = getTickCount();
            Ptr<SparseOpticalFlow> algorithm = SparsePyrLKOpticalFlow::create();
            times[2] = getTickCount();
            algorithm->calc(img1, img2, prevPts, nextPts, status);
            for (size_t i = 0; i < prevPts.size(); i++)
                flw.at<Point2f>(prevPts[i]) = status[i] ? nextPts[i] - prevPts[i] : Point2f();
            times[3] = getTickCount();
        }
        break;
    default:
        return vector<double>();
    }
    vector<double> vtime;
    vtime.push_back((times[1] - times[0]) * 1000. / getTickFrequency());
    vtime.push_back((times[2] - times[1]) * 1000. / getTickFrequency());
    vtime.push_back((times[3] - times[2]) * 1000. / getTickFrequency());
    return vtime;
}

vector<double> evalFlow(const Mat &flw, const Mat &refflw, const Mat &mask)
{
    vector<double> retval;
    Mat delta;
    absdiff(flw, refflw, delta);
    Mat mflw, merr;
    Mat uv[2];
    split(refflw, uv);
    magnitude(uv[0], uv[1], mflw);
    split(delta, uv);
    magnitude(uv[0], uv[1], merr);

    int ptCount = countNonZero(mask);
    retval.push_back(norm(merr, NORM_INF, mask));
    retval.push_back(norm(merr, NORM_L2, mask));
    retval.push_back(norm(merr, NORM_L1, mask) / ptCount);
    retval.push_back(norm(merr, NORM_L2, mask) / sqrt(ptCount));

    mflw *= -0.05;
    threshold(mflw, mflw, -3, -3, THRESH_TRUNC);
    mflw = 0 - mflw;
    Mat clear_mask = (merr >= mflw) & mask;

    retval.push_back((double)(countNonZero(clear_mask)) / ptCount);
    retval.push_back(norm(merr, NORM_INF, clear_mask));
    retval.push_back(norm(merr, NORM_L2, clear_mask));
    retval.push_back(norm(merr, NORM_L1, clear_mask) / ptCount);
    retval.push_back(norm(merr, NORM_L2, clear_mask) / sqrt(ptCount));

    return retval;
}

void evalAcc(vector<double> &norm, vector<double> sample)
{
    norm[0] = max(norm[0], sample[0]);
    norm[1] += sample[0];
    norm[2] += sample[1];
    norm[3] += sample[2];
    norm[4] += sample[3];

    norm[5] += sample[4];
    norm[6] = max(norm[2], sample[5]);
    norm[7] += sample[5];
    norm[8] += sample[6];
    norm[9] += sample[7];
    norm[10] += sample[8];
}

}

#define SAMPLE_COUNT 200
//#define SAMPLE_COUNT 2
#define FLOW_COUNT 11

int main(int, char**)
{
    vector< vector<double> > time(FLOW_COUNT, vector<double>(3, 0));
    vector<int> processedSamples(FLOW_COUNT, 0);
    vector< vector<double> > normO(FLOW_COUNT, vector<double>(11, 0));
    vector< vector<double> > normN(FLOW_COUNT, vector<double>(11, 0));
    for (int im_id = 0; im_id < SAMPLE_COUNT; im_id++)
    {
        Mat img1, img2, flwn, flwo, flwn_mask, flwo_mask;
        if (!getSample(im_id, img1, img2, flwn, flwo, flwn_mask, flwo_mask))
            continue;
        for (int flow_id = 0; flow_id < FLOW_COUNT; flow_id++)
        {
            Mat flw;
            vector<double> frameTimes = getFlow(flow_id, img1.clone(), img2.clone(), flw);
            if (frameTimes.empty())
                continue;
            time[flow_id][0] += frameTimes[0];
            time[flow_id][1] += frameTimes[1];
            time[flow_id][2] += frameTimes[2];
            processedSamples[flow_id]++;
            flowShow(flw, flwo, flwo_mask, 0);
            evalAcc(normO[flow_id], evalFlow(flw, flwo, flwo_mask));
            evalAcc(normN[flow_id], evalFlow(flw, flwn, flwn_mask));
        }
        printf("flowid;time1;time2;time3;normO1;normO2;normO3;normO4;normO5;normO6;normO7;normO8;normO9;normO10;normO11;normN1;normN2;normN3;normN4;normN5;normN6;normN7;normN8;normN9;normN10;normN11;\n");
        for (int flow_id = 0; flow_id < FLOW_COUNT; flow_id++)
        {
            time[flow_id][0] /= processedSamples[flow_id];
            time[flow_id][1] /= processedSamples[flow_id];
            time[flow_id][2] /= processedSamples[flow_id];

            normO[flow_id][ 1] /= processedSamples[flow_id];
            normO[flow_id][ 2] /= processedSamples[flow_id];
            normO[flow_id][ 3] /= processedSamples[flow_id];
            normO[flow_id][ 4] /= processedSamples[flow_id];
            normO[flow_id][ 5] /= processedSamples[flow_id];
            normO[flow_id][ 7] /= processedSamples[flow_id];
            normO[flow_id][ 8] /= processedSamples[flow_id];
            normO[flow_id][ 9] /= processedSamples[flow_id];
            normO[flow_id][10] /= processedSamples[flow_id];
            normN[flow_id][ 1] /= processedSamples[flow_id];
            normN[flow_id][ 2] /= processedSamples[flow_id];
            normN[flow_id][ 3] /= processedSamples[flow_id];
            normN[flow_id][ 4] /= processedSamples[flow_id];
            normN[flow_id][ 5] /= processedSamples[flow_id];
            normN[flow_id][ 7] /= processedSamples[flow_id];
            normN[flow_id][ 8] /= processedSamples[flow_id];
            normN[flow_id][ 9] /= processedSamples[flow_id];
            normN[flow_id][10] /= processedSamples[flow_id];
            printf("%d;%f;%f;%f;%f;%f;%f;%f;%f;%f;%f;%f;%f;%f;%f;%f;%f;%f;%f;%f;%f;%f;%f;%f;%f;%f;\n",
                   flow_id, time[flow_id][0], time[flow_id][1], time[flow_id][2],
                   normO[flow_id][0], normO[flow_id][1], normO[flow_id][2], normO[flow_id][3], normO[flow_id][4], normO[flow_id][5], normO[flow_id][6], normO[flow_id][7], normO[flow_id][8], normO[flow_id][9], normO[flow_id][10],
                   normN[flow_id][0], normN[flow_id][1], normN[flow_id][2], normN[flow_id][3], normN[flow_id][4], normN[flow_id][5], normN[flow_id][6], normN[flow_id][7], normN[flow_id][8], normN[flow_id][9], normN[flow_id][10]);
        }
    }
    return 0;
}
