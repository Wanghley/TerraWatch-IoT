#pragma once
#include <cstdio>
#include "doppler.h"  // your DopplerData struct
#include "thermal.h"  // your ThermalFrame struct

// --- Configuration ---
#ifndef AMG88xx_PIXEL_ARRAY_SIZE
#define AMG88xx_PIXEL_ARRAY_SIZE 64
#endif

// --- Data Structures ---
struct Features {
    int label;

    // Doppler
    float doppler_speed;
    float doppler_range;
    float doppler_energy;

    // Mic
    float mic_rms_mean;
    float mic_peak_mean;

    // Mic RMS stats
    float mic_rms_samples_mean;
    float mic_rms_samples_std;
    float mic_rms_samples_min;
    float mic_rms_samples_max;

    // Thermal basic stats
    float thermal_mean;
    float thermal_std;
    float thermal_min;
    float thermal_max;

    // Thermal spatial features
    float thermal_centroid_y;
    float thermal_centroid_x;
    float thermal_vertical_diff;
    float thermal_horizontal_diff;
};

// --- Feature Extraction Module ---
namespace FeatureExtractor {

    // Compute mean, std, min, max
    void computeStats(const float* data, int n, float& mean, float& stddev, float& minVal, float& maxVal);

    // Extract all features
    Features extractFeatures(int label, const DopplerData& doppler, 
                             const ThermalFrame& thermal, 
                             const float* micRmsSamples, int numMicSamples,
                             float micRmsMean, float micPeakMean);

    // Print features for debugging
    void printFeatures(const Features& f);
}
