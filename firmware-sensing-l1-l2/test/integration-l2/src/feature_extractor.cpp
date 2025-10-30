#include "feature_extractor.h"
#include <cmath>

namespace FeatureExtractor {

void computeStats(const float* data, int n, float& mean, float& stddev, float& minVal, float& maxVal) {
    if (n == 0) {
        mean = stddev = minVal = maxVal = 0.0f;
        return;
    }

    minVal = maxVal = data[0];
    float sum = 0.0f;
    float sumSq = 0.0f;

    for (int i = 0; i < n; i++) {
        float val = data[i];
        sum += val;
        sumSq += val * val;
        if (val < minVal) minVal = val;
        if (val > maxVal) maxVal = val;
    }

    mean = sum / static_cast<float>(n);
    float variance = (sumSq / static_cast<float>(n)) - (mean * mean);
    if (variance < 0.0f) variance = 0.0f;
    stddev = std::sqrt(variance);
}

Features extractFeatures(int label, const DopplerData& doppler, 
                         const ThermalFrame& thermal, 
                         const float* micRmsSamples, int numMicSamples,
                         float micRmsMean, float micPeakMean) {
    Features features;
    features.label = label;

    // Doppler
    features.doppler_speed = doppler.speed;
    features.doppler_range = doppler.range;
    features.doppler_energy = doppler.energy;

    // Mic
    features.mic_rms_mean = micRmsMean;
    features.mic_peak_mean = micPeakMean;

    // Mic RMS stats
    if (numMicSamples > 0) {
        computeStats(micRmsSamples, numMicSamples,
                     features.mic_rms_samples_mean,
                     features.mic_rms_samples_std,
                     features.mic_rms_samples_min,
                     features.mic_rms_samples_max);
    } else {
        features.mic_rms_samples_mean = 0.0f;
        features.mic_rms_samples_std = 0.0f;
        features.mic_rms_samples_min = 0.0f;
        features.mic_rms_samples_max = 0.0f;
    }

    // Thermal
    if (thermal.pixels != nullptr) {
        computeStats(thermal.pixels, AMG88xx_PIXEL_ARRAY_SIZE,
                     features.thermal_mean,
                     features.thermal_std,
                     features.thermal_min,
                     features.thermal_max);

        // Centroid
        float totalWeight = 0.0f;
        float weightedY = 0.0f;
        float weightedX = 0.0f;

        for (int y = 0; y < 8; y++) {
            for (int x = 0; x < 8; x++) {
                int idx = y * 8 + x;
                float weight = thermal.pixels[idx] - features.thermal_min;
                totalWeight += weight;
                weightedY += weight * static_cast<float>(y);
                weightedX += weight * static_cast<float>(x);
            }
        }

        if (totalWeight > 0.0f) {
            features.thermal_centroid_y = weightedY / totalWeight;
            features.thermal_centroid_x = weightedX / totalWeight;
        } else {
            features.thermal_centroid_y = 3.5f;
            features.thermal_centroid_x = 3.5f;
        }

        // Half-image differences
        float topSum = 0.0f, bottomSum = 0.0f, leftSum = 0.0f, rightSum = 0.0f;
        for (int y = 0; y < 8; y++) {
            for (int x = 0; x < 8; x++) {
                int idx = y * 8 + x;
                float val = thermal.pixels[idx];
                if (y < 4) topSum += val; else bottomSum += val;
                if (x < 4) leftSum += val; else rightSum += val;
            }
        }

        features.thermal_vertical_diff = (topSum / 32.0f) - (bottomSum / 32.0f);
        features.thermal_horizontal_diff = (leftSum / 32.0f) - (rightSum / 32.0f);

    } else {
        features.thermal_mean = 0.0f;
        features.thermal_std = 0.0f;
        features.thermal_min = 0.0f;
        features.thermal_max = 0.0f;
        features.thermal_centroid_y = 3.5f;
        features.thermal_centroid_x = 3.5f;
        features.thermal_vertical_diff = 0.0f;
        features.thermal_horizontal_diff = 0.0f;
    }

    return features;
}

void printFeatures(const Features& f) {
    printf("--- Extracted Features ---\n");
    printf("Label: %d\n", f.label);
    printf("Doppler - speed: %.2f, range: %.2f, energy: %.2f\n",
           f.doppler_speed, f.doppler_range, f.doppler_energy);
    printf("Mic - rms_mean: %.4f, peak_mean: %.2f\n",
           f.mic_rms_mean, f.mic_peak_mean);
    printf("Mic RMS - mean: %.4f, std: %.4f, min: %.4f, max: %.4f\n",
           f.mic_rms_samples_mean, f.mic_rms_samples_std,
           f.mic_rms_samples_min, f.mic_rms_samples_max);
    printf("Thermal - mean: %.2f, std: %.2f, min: %.2f, max: %.2f\n",
           f.thermal_mean, f.thermal_std, f.thermal_min, f.thermal_max);
    printf("Thermal Centroid - y: %.2f, x: %.2f\n",
           f.thermal_centroid_y, f.thermal_centroid_x);
    printf("Thermal Diff - vertical: %.2f, horizontal: %.2f\n",
           f.thermal_vertical_diff, f.thermal_horizontal_diff);
}

} // namespace FeatureExtractor
