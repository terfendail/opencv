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
    if (waitTime)
    {
        Mat showImg(flow.rows*3, flow.cols, flow.type());
        flow.copyTo(showImg(Rect(0, 0, flow.cols, flow.rows)));
        showImg(Rect(0, 0, flow.cols, flow.rows)).setTo(Scalar(0, 0), 255 - mask);
        showImg(Rect(0, 0, flow.cols, flow.rows)).copyTo(showImg(Rect(0, 2 * flow.rows, flow.cols, flow.rows)));
        refflow.copyTo(showImg(Rect(0, flow.rows, flow.cols, flow.rows)));
        showImg(Rect(0, 2 * flow.rows, flow.cols, flow.rows)) -= refflow;
        imshow("error", convertFlow(showImg));
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

pair< String, vector<double> > getDISFlow(int method, Mat img1, Mat img2, Mat& flw)
{
    String metname;
    int64 times[4];
    flw = Mat(img1.size(), CV_32FC2);

    int fscale[] = { 0, 1, 2, 3 };
    int gditer[] = { 256, 128, 96, 64, 32, 16, 12, 8 };
    pair<int, int> psovsize[] = { pair<int, int>(16, 15), pair<int, int>(16, 12), pair<int, int>(16, 8), pair<int, int>(16, 6), pair<int, int>(16, 5), pair<int, int>(16, 4), pair<int, int>(16, 3), pair<int, int>(16, 2),
                                  pair<int, int>(12, 11), pair<int, int>(12, 9), pair<int, int>(12, 8), pair<int, int>(12, 6), pair<int, int>(12, 5), pair<int, int>(12, 4), pair<int, int>(12, 3), pair<int, int>(12, 2),
                                  pair<int, int>(8, 7), pair<int, int>(8, 6), pair<int, int>(8, 4), pair<int, int>(8, 3), pair<int, int>(8, 2), pair<int, int>(4, 3), pair<int, int>(4, 2), pair<int, int>(4, 1) };
    int refiter[] = { 16, 12, 10, 8, 6, 5, 4, 3 };
    float refalfa[] = { 20.f, 10.f, 5.f, 1.f };
    float refgama[] = { 20.f, 10.f, 5.f, 1.f };
    float refdelt[] = { 20.f, 10.f, 5.f, 1.f };

    int fscale_id = method % (sizeof(fscale) / sizeof(fscale[0]) );
    method /= sizeof(fscale) / sizeof(fscale[0]);
    int gditer_id = method % (sizeof(gditer) / sizeof(gditer[0]));
    method /= sizeof(gditer) / sizeof(gditer[0]);
    int psovsize_id = method % (sizeof(psovsize) / sizeof(psovsize[0]));
    method /= sizeof(psovsize) / sizeof(psovsize[0]);
    int refiter_id = method % (sizeof(refiter) / sizeof(refiter[0]));
    method /= sizeof(refiter) / sizeof(refiter[0]);
    int refalfa_id = method % (sizeof(refalfa) / sizeof(refalfa[0]));
    method /= sizeof(refalfa) / sizeof(refalfa[0]);
    int refgama_id = method % (sizeof(refgama) / sizeof(refgama[0]));
    method /= sizeof(refgama) / sizeof(refgama[0]);
    int refdelt_id = method % (sizeof(refdelt) / sizeof(refdelt[0]));
    method /= sizeof(refdelt) / sizeof(refdelt[0]);
    int meannorm = method % 2;
    method /= 2;
    int spatprop = method % 2;
    method /= 2;

    if(method)
        return pair< String, vector<double> >(String(), vector<double>());

    metname = format("DISflow_par_fs%d_it%d_pss%d_%d_rit%d_a%f_g%f_d%f_mn%d_sp%d", fscale[fscale_id], gditer[gditer_id], psovsize[psovsize_id].first, psovsize[psovsize_id].second,
                                                                                   refiter[refiter_id], refalfa[refalfa_id], refgama[refgama_id], refdelt[refdelt_id], meannorm, spatprop);

    times[0] = getTickCount();
    if (img1.channels() == 3)
    {
        cvtColor(img1, img1, COLOR_BGR2GRAY);
        cvtColor(img2, img2, COLOR_BGR2GRAY);
    }
    times[1] = getTickCount();
    Ptr<optflow::DISOpticalFlow> DIS = optflow::createOptFlow_DIS(optflow::DISOpticalFlow::PRESET_FAST);
    DIS->setFinestScale(fscale[fscale_id]);
    DIS->setGradientDescentIterations(gditer[gditer_id]);
    DIS->setPatchSize(psovsize[psovsize_id].first);
    DIS->setPatchStride(psovsize[psovsize_id].second);
    DIS->setUseMeanNormalization((bool)meannorm);
    DIS->setUseSpatialPropagation((bool)spatprop);
    DIS->setVariationalRefinementIterations(refiter[refiter_id]);
    DIS->setVariationalRefinementAlpha(refalfa[refalfa_id]);
    DIS->setVariationalRefinementDelta(refgama[refgama_id]);
    DIS->setVariationalRefinementGamma(refdelt[refdelt_id]);
    Ptr<DenseOpticalFlow> algorithm = DIS;
    times[2] = getTickCount();
    algorithm->calc(img1, img2, flw);
    times[3] = getTickCount();

    vector<double> vtime;
    vtime.push_back((times[1] - times[0]) * 1000. / getTickFrequency());
    vtime.push_back((times[2] - times[1]) * 1000. / getTickFrequency());
    vtime.push_back((times[3] - times[2]) * 1000. / getTickFrequency());
    return pair< String, vector<double> >(metname, vtime);
}


pair< String, vector<double> > getFlow(int method, Mat img1, Mat img2, Mat& flw)
{
    String metname;
    int64 times[4];
    flw = Mat(img1.size(), CV_32FC2);
    switch (method)
    {
    case 0:
        metname = "sparsetodenseflow";
        {
            times[0] = times[1] = getTickCount();
            Ptr<DenseOpticalFlow> algorithm = optflow::createOptFlow_SparseToDense();
            times[2] = getTickCount();
            algorithm->calc(img1, img2, flw);
            times[3] = getTickCount();
        }
        break;
    case 1:
        metname = "farneback";
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
    case 2:
        metname = "simpleflow";
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
    case 3:
        metname = "pcaflow";
        {
            times[0] = times[1] = getTickCount();
            Ptr<DenseOpticalFlow> algorithm = optflow::createOptFlow_PCAFlow();
            times[2] = getTickCount();
            algorithm->calc(img1, img2, flw);
            times[3] = getTickCount();
        }
        break;
    case 4:
        metname = "deepflow";
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
    case 5:
        metname = "DISflow_ultrafast";
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
    case 6:
        metname = "DISflow_fast";
        {
            times[0] = getTickCount();
            if (img1.channels() == 3)
            {
                cvtColor(img1, img1, COLOR_BGR2GRAY);
                cvtColor(img2, img2, COLOR_BGR2GRAY);
            }
            times[1] = getTickCount();
            Ptr<optflow::DISOpticalFlow> DIS = optflow::createOptFlow_DIS(optflow::DISOpticalFlow::PRESET_FAST);
            //DIS->setFinestScale(2);
            //DIS->setGradientDescentIterations(16);
            //DIS->setPatchSize(8);
            //DIS->setPatchStride(4);
            //DIS->setUseMeanNormalization(true);
            //DIS->setUseSpatialPropagation(true);
            //DIS->setVariationalRefinementIterations(5);
            //DIS->setVariationalRefinementAlpha(20.f);
            //DIS->setVariationalRefinementDelta(5.f);
            //DIS->setVariationalRefinementGamma(10.f);
            Ptr<DenseOpticalFlow> algorithm = DIS;
            times[2] = getTickCount();
            algorithm->calc(img1, img2, flw);
            times[3] = getTickCount();
        }
        break;
    case 7:
        metname = "DISflow_medium";
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

    case 8:
        metname = "DISflow_05";
        {
            times[0] = getTickCount();
            if (img1.channels() == 3)
            {
                cvtColor(img1, img1, COLOR_BGR2GRAY);
                cvtColor(img2, img2, COLOR_BGR2GRAY);
            }
            times[1] = getTickCount();
            Ptr<optflow::DISOpticalFlow> DIS = optflow::createOptFlow_DIS(optflow::DISOpticalFlow::PRESET_FAST);
            DIS->setFinestScale(0);
            DIS->setGradientDescentIterations(256);
            DIS->setPatchSize(12);
            DIS->setPatchStride(9);
            DIS->setUseMeanNormalization(true);
            DIS->setUseSpatialPropagation(true);
            DIS->setVariationalRefinementIterations(5);
            DIS->setVariationalRefinementAlpha(10.f);
            DIS->setVariationalRefinementDelta(5.f);
            DIS->setVariationalRefinementGamma(10.f);
            Ptr<DenseOpticalFlow> algorithm = DIS;
            times[2] = getTickCount();
            algorithm->calc(img1, img2, flw);
            times[3] = getTickCount();
        }
        break;
    case 9:
        metname = "DISflow_10";
        {
            times[0] = getTickCount();
            if (img1.channels() == 3)
            {
                cvtColor(img1, img1, COLOR_BGR2GRAY);
                cvtColor(img2, img2, COLOR_BGR2GRAY);
            }
            times[1] = getTickCount();
            Ptr<optflow::DISOpticalFlow> DIS = optflow::createOptFlow_DIS(optflow::DISOpticalFlow::PRESET_FAST);
            DIS->setFinestScale(1);
            DIS->setGradientDescentIterations(16);
            DIS->setPatchSize(12);
            DIS->setPatchStride(9);
            DIS->setUseMeanNormalization(true);
            DIS->setUseSpatialPropagation(true);
            DIS->setVariationalRefinementIterations(5);
            DIS->setVariationalRefinementAlpha(10.f);
            DIS->setVariationalRefinementDelta(5.f);
            DIS->setVariationalRefinementGamma(10.f);
            Ptr<DenseOpticalFlow> algorithm = DIS;
            times[2] = getTickCount();
            algorithm->calc(img1, img2, flw);
            times[3] = getTickCount();
        }
        break;
    case 10:
        metname = "DISflow_300";
        {
            times[0] = getTickCount();
            if (img1.channels() == 3)
            {
                cvtColor(img1, img1, COLOR_BGR2GRAY);
                cvtColor(img2, img2, COLOR_BGR2GRAY);
            }
            times[1] = getTickCount();
            Ptr<optflow::DISOpticalFlow> DIS = optflow::createOptFlow_DIS(optflow::DISOpticalFlow::PRESET_FAST);
            DIS->setFinestScale(3);
            DIS->setGradientDescentIterations(12);
            DIS->setPatchSize(8);
            DIS->setPatchStride(4);
            DIS->setUseMeanNormalization(true);
            DIS->setUseSpatialPropagation(true);
            DIS->setVariationalRefinementIterations(5);
            DIS->setVariationalRefinementAlpha(10.f);
            DIS->setVariationalRefinementDelta(5.f);
            DIS->setVariationalRefinementGamma(10.f);
            Ptr<DenseOpticalFlow> algorithm = DIS;
            times[2] = getTickCount();
            algorithm->calc(img1, img2, flw);
            times[3] = getTickCount();
        }
        break;
    case 11:
        metname = "DISflow_600";
        {
            times[0] = getTickCount();
            if (img1.channels() == 3)
            {
                cvtColor(img1, img1, COLOR_BGR2GRAY);
                cvtColor(img2, img2, COLOR_BGR2GRAY);
            }
            times[1] = getTickCount();
            Ptr<optflow::DISOpticalFlow> DIS = optflow::createOptFlow_DIS(optflow::DISOpticalFlow::PRESET_FAST);
            DIS->setFinestScale(3);
            DIS->setGradientDescentIterations(16);
            DIS->setPatchSize(8);
            DIS->setPatchStride(3);
            DIS->setUseMeanNormalization(true);
            DIS->setUseSpatialPropagation(true);
            DIS->setVariationalRefinementIterations(5);
            DIS->setVariationalRefinementAlpha(10.f);
            DIS->setVariationalRefinementDelta(5.f);
            DIS->setVariationalRefinementGamma(10.f);
            Ptr<DenseOpticalFlow> algorithm = DIS;
            times[2] = getTickCount();
            algorithm->calc(img1, img2, flw);
            times[3] = getTickCount();
        }
        break;

    default:
        return pair< String, vector<double> >(String(), vector<double>());
    }
    vector<double> vtime;
    vtime.push_back((times[1] - times[0]) * 1000. / getTickFrequency());
    vtime.push_back((times[2] - times[1]) * 1000. / getTickFrequency());
    vtime.push_back((times[3] - times[2]) * 1000. / getTickFrequency());
    return pair< String, vector<double> >(metname, vtime);
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

int main(int, char**)
{
    vector< vector<double> > time;
    vector<int> processedSamples;
    vector< String > names;
    vector< vector<double> > normO;
    vector< vector<double> > normN;
    for (int im_id = 0; im_id < SAMPLE_COUNT; im_id++)
    {
        Mat img1, img2, flwn, flwo, flwn_mask, flwo_mask;
        if (!getSample(im_id, img1, img2, flwn, flwo, flwn_mask, flwo_mask))
            continue;
        for (int flow_id = 0; true; flow_id++)
        {
            Mat flw;
            pair< String, vector<double> > flowEv = getDISFlow(flow_id, img1.clone(), img2.clone(), flw);
            if (flowEv.second.empty())
                break;
            printf("Evaluated %s\n", flowEv.first.c_str());

            while(time.size() <= flow_id)
                time.push_back(vector<double>(3, 0));
            while (processedSamples.size() <= flow_id)
                processedSamples.push_back(0);
            while (normO.size() <= flow_id)
                normO.push_back(vector<double>(11, 0));
            while (normN.size() <= flow_id)
                normN.push_back(vector<double>(11, 0));
            while (names.size() <= flow_id)
                names.push_back(String());
            if (names[flow_id].empty())
                names[flow_id] = flowEv.first;

            time[flow_id][0] += flowEv.second[0];
            time[flow_id][1] += flowEv.second[1];
            time[flow_id][2] += flowEv.second[2];
            processedSamples[flow_id]++;
            flowShow(flw, flwo, flwo_mask, 0);
            evalAcc(normO[flow_id], evalFlow(flw, flwo, flwo_mask));
            evalAcc(normN[flow_id], evalFlow(flw, flwn, flwn_mask));
        }
        printf("flowid;time1;time2;time3;normO1;normO2;normO3;normO4;normO5;normO6;normO7;normO8;normO9;normO10;normO11;normN1;normN2;normN3;normN4;normN5;normN6;normN7;normN8;normN9;normN10;normN11;\n");
        for (int flow_id = 0; flow_id < time.size(); flow_id++)
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
            printf("%s;%f;%f;%f;%f;%f;%f;%f;%f;%f;%f;%f;%f;%f;%f;%f;%f;%f;%f;%f;%f;%f;%f;%f;%f;%f;\n",
                   names[flow_id].c_str(), time[flow_id][0], time[flow_id][1], time[flow_id][2],
                   normO[flow_id][0], normO[flow_id][1], normO[flow_id][2], normO[flow_id][3], normO[flow_id][4], normO[flow_id][5], normO[flow_id][6], normO[flow_id][7], normO[flow_id][8], normO[flow_id][9], normO[flow_id][10],
                   normN[flow_id][0], normN[flow_id][1], normN[flow_id][2], normN[flow_id][3], normN[flow_id][4], normN[flow_id][5], normN[flow_id][6], normN[flow_id][7], normN[flow_id][8], normN[flow_id][9], normN[flow_id][10]);
        }
    }
    return 0;
}
