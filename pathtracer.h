#ifndef PATHTRACER_H
#define PATHTRACER_H

#include <QImage>
#include <random>

#include "scene/scene.h"

struct Settings {
    int samplesPerPixel;
    bool directLightingOnly; // if true, ignore indirect lighting
    int numDirectLightingSamples; // number of shadow rays to trace from each intersection point
    float pathContinuationProb; // probability of spawning a new secondary ray == (1-pathTerminationProb)
};

class PathTracer
{
public:
    PathTracer(int width, int height);

    void traceScene(QRgb *imageData, const Scene &scene);
    Settings settings;

private:
    int m_width, m_height;

    std::mt19937 gen;
    std::random_device rd;
    std::uniform_real_distribution<> dis;

    void toneMap(QRgb *imageData, std::vector<Eigen::Vector3f> &intensityValues);

    Eigen::Vector3f tracePixel(int x, int y, const Scene &scene, const Eigen::Matrix4f &invViewMatrix);
    Eigen::Vector3f traceRay(const Ray& r, const Scene &scene, bool countEmission, float pdf_rr);

    Eigen::Vector3f sampleNextDir(Eigen::Vector3f N);
    Eigen::Vector3f sampleOnTriangle(const Eigen::Vector3<Eigen::Vector3f>& vertices);

    Eigen::Vector3f directLighting(const Scene& scene, const Ray& ray, const Eigen::Vector3f& prevHit, const Eigen::Vector3f& prevN, Eigen::Vector3f& w_i);

    float schlicksApprox(float ior_i, float ior_t, float cos_theta_i);
    float getTriangleArea(const Eigen::Vector3<Eigen::Vector3f>& vertices);
    float getHaltonSequenceNum(int index, int base);
};

#endif // PATHTRACER_H
