#include "pathtracer.h"
#include "QtCore/qmath.h"

#include <iostream>
#include <random>
#include <cmath>

#include <Eigen/Dense>

#include <util/CS123Common.h>

using namespace Eigen;

PathTracer::PathTracer(int width, int height)
    : m_width(width), m_height(height), dis(0, 1), gen(rd())
{
}

void PathTracer::traceScene(QRgb *imageData, const Scene& scene)
{
    std::vector<Vector3f> intensityValues(m_width * m_height);
    Matrix4f invViewMat = (scene.getCamera().getScaleMatrix() * scene.getCamera().getViewMatrix()).inverse();
    for(int y = 0; y < m_height; ++y) {
//        #pragma omp parallel for
        for(int x = 0; x < m_width; ++x) {
            int offset = x + (y * m_width);
            intensityValues[offset] = tracePixel(x, y, scene, invViewMat);
        }
    }

//    outputPFM("/Users/feiyue/test.pfm", m_width, m_height, intensityValues);
    toneMap(imageData, intensityValues);
}

Vector3f PathTracer::tracePixel(int x, int y, const Scene& scene, const Matrix4f &invViewMatrix)
{
    Vector3f p(0, 0, 0);
    Vector3f light(0, 0, 0);

    //// RANDOM SAMPLING

//    Vector3f d((2.f * x / m_width) - 1, 1 - (2.f * y / m_height), -1);
//    d.normalize();

//    Ray r(p, d);
//    r = r.transform(invViewMatrix);

//    for (int i = 0; i < settings.samplesPerPixel; i++) {
//        light += traceRay(r, scene, true, settings.pathContinuationProb);
//    }
//    return light / settings.samplesPerPixel;

    //// STRATIFIED SAMPLING

//    // square root of samplesPerPixel
//    int sqrtSpp = sqrt(settings.samplesPerPixel);

//    for (int i = 0; i < sqrtSpp; i++) {
//        for (int j = 0; j < sqrtSpp; j++) {
//            float offsetX = (i + dis(gen)) / sqrtSpp;
//            float offsetY = (j + dis(gen)) / sqrtSpp;

//            float u = x + offsetX;
//            float v = y + offsetY;

//            Vector3f d((2.f * u / m_width) - 1, 1 - (2.f * v / m_height), -1);
//            d.normalize();

//            Ray r(p, d);
//            r = r.transform(invViewMatrix);

//            light += traceRay(r, scene, true, settings.pathContinuationProb);
//        }
//    }
//    return light / (sqrtSpp * sqrtSpp);

    //// LOW DISCREPANCY SAMPLING (QMC)

    // prime number bases
    const int baseX = 2;
    const int baseY = 3;

    for (int i = 0; i < settings.samplesPerPixel; i++) {
        float offsetX = getHaltonSequenceNum(i + 1, baseX);
        float offsetY = getHaltonSequenceNum(i + 1, baseY);

        float u = x + offsetX;
        float v = y + offsetY;

        Vector3f d((2.f * u / m_width) - 1, 1 - (2.f * v / m_height), -1);
        d.normalize();

        Ray r(p, d);
        r = r.transform(invViewMatrix);

        light += traceRay(r, scene, true, settings.pathContinuationProb);
    }
    return light / settings.samplesPerPixel;
}

float PathTracer::getHaltonSequenceNum(int index, int base)
{
    float res = 0.f;
    float f = 1.f / base;
    int i = index;

    while (i > 0) {
        res += f * (i % base);
        i /= base;
        f /= base;
    }
    return res;
}

Vector3f PathTracer::traceRay(const Ray& r, const Scene& scene, bool countEmission, float pdf_rr)
{
    IntersectionInfo i;
    Ray ray(r);
    Vector3f l(0, 0, 0);

    if (scene.getIntersection(ray, &i)) {
        const Triangle *t = static_cast<const Triangle *>(i.data);
        const tinyobj::material_t& mat = t->getMaterial();

        Vector3f N = t->getNormal(i); // surface normal at intersection

        //// DIRECT LIGHTING
        Vector3f direct_w_i(0, 0, 0);
        Vector3f direct_rad = directLighting(scene, ray, i.hit, N, direct_w_i);

        if (direct_rad != Vector3f(0, 0, 0)) {
            if (mat.illum == 2) {
                // glossy
                if (Vector3f(mat.specular) != Vector3f(0, 0, 0)) {
                    Vector3f refl = 2.f * std::max(-ray.d.dot(N), 0.f) * N + ray.d;
                    refl = refl.normalized();

                    float dot_prod_term = refl.dot(direct_w_i);
                    Vector3f glossy_brdf(mat.specular);
                    glossy_brdf = glossy_brdf * (mat.shininess + 2) / (2 * M_PI) * std::pow(dot_prod_term, mat.shininess);

                    l += direct_rad.cwiseProduct(glossy_brdf) * direct_w_i.dot(N);
                }
                // diffuse
                else {
                    Vector3f diffuse_brdf(mat.diffuse);
                    diffuse_brdf = diffuse_brdf / M_PI;

                    l += direct_rad.cwiseProduct(diffuse_brdf) * direct_w_i.dot(N);
                }
            }
        }

        //// INDIRECT LIGHTING
        if (dis(gen) < pdf_rr && !settings.directLightingOnly) {
            float pdf = 1.f / (2.f * M_PI);
            Vector3f w_i = sampleNextDir(N); // incident ray
            Ray r_i(i.hit, w_i);

            if (mat.illum == 2) {
                // glossy
                if (Vector3f(mat.specular) != Vector3f(0, 0, 0)) {
                    Vector3f refl = 2.f * std::max(-ray.d.dot(N), 0.f) * N + ray.d;
                    refl = refl.normalized();

                    float dot_prod_term = refl.dot(w_i);
                    Vector3f glossy_brdf(mat.specular);
                    glossy_brdf = glossy_brdf * (mat.shininess + 2) / (2 * M_PI) * std::pow(dot_prod_term, mat.shininess);

                    Vector3f radiance = traceRay(r_i, scene, false, glossy_brdf.maxCoeff());
                    l += radiance.cwiseProduct(glossy_brdf) * w_i.dot(N) / (pdf * pdf_rr);
                }
                // diffuse
                else {
                    Vector3f diffuse_brdf(mat.diffuse);
                    diffuse_brdf = diffuse_brdf / M_PI;

                    Vector3f radiance = traceRay(r_i, scene, false, diffuse_brdf.maxCoeff());
                    l += radiance.cwiseProduct(diffuse_brdf) * w_i.dot(N) / (pdf * pdf_rr);
                }
            }
            // mirror/ideal specular
            else if (mat.illum == 5) {
                Vector3f refl = 2.f * std::max(-ray.d.dot(N), 0.f) * N + ray.d;
                refl = refl.normalized();

                r_i = Ray(i.hit, refl);
                Vector3f radiance = traceRay(r_i, scene, true, pdf_rr);
                l += radiance / pdf_rr;
            }
            else if (mat.illum == 7) {
                float ior_ratio = 1.f / mat.ior; // assume air-to-glass
                float cos_theta_i = -ray.d.dot(N); // assume air-to-glass

                bool isAirToGlass = cos_theta_i >= 0;

                // glass-to-air
                if (!isAirToGlass) {
                    cos_theta_i = -cos_theta_i;
                    ior_ratio = 1.f / ior_ratio;
                    N = -N;
                }

                float R_theta_i = isAirToGlass ?
                                  schlicksApprox(1, mat.ior, cos_theta_i) :
                                  schlicksApprox(mat.ior, 1, cos_theta_i);

                // mirror/ideal specular
                if (dis(gen) < R_theta_i) {
                    Vector3f refl = 2.f * std::max(-ray.d.dot(N), 0.f) * N + ray.d;
                    refl = refl.normalized();

                    r_i = Ray(i.hit, refl);
                    Vector3f radiance = traceRay(r_i, scene, true, pdf_rr);
                    l += radiance / pdf_rr;
                }
                // refractive
                else {
                    float cos_theta_t_sq = 1 - pow(ior_ratio, 2) * (1.f - pow(cos_theta_i, 2));

                    if (cos_theta_t_sq > 0) {
                        //// W/O ATTENUATION
//                        float cos_theta_t = sqrt(cos_theta_t_sq);
//                        Vector3f refract = ior_ratio * ray.d + (ior_ratio * cos_theta_i - cos_theta_t) * N;
//                        refract = refract.normalized();

//                        r_i = Ray(i.hit, refract);
//                        Vector3f radiance = traceRay(r_i, scene, true, pdf_rr);
//                        l += radiance / pdf_rr;

                        //// ATTENUATE REFRACTED PATHS

                        float cos_theta_t = sqrt(cos_theta_t_sq);
                        Vector3f refract = ior_ratio * ray.d + (ior_ratio * cos_theta_i - cos_theta_t) * N;
                        refract = refract.normalized();

                        IntersectionInfo i_refrac;
                        r_i = Ray(i.hit, refract);

                        if (scene.getIntersection(r_i, &i_refrac)) {
                            float distance = (i_refrac.hit - i.hit).norm();

                            // beer's law
                            Vector3f attenuation(exp(-distance * mat.transmittance[0]),
                                                 exp(-distance * mat.transmittance[1]),
                                                 exp(-distance * mat.transmittance[2]));

                            Vector3f radiance = traceRay(r_i, scene, true, pdf_rr);
                            l += radiance.cwiseProduct(attenuation) / pdf_rr;
                        }
                    }
                }
            }
        }

        // add emission for ideal specular
        if (countEmission) {
            Vector3f le(mat.emission);
            l += le;
        }
    }
    return l;
}

float PathTracer::schlicksApprox(float ior_i, float ior_t, float cos_theta_i)
{
    float R_0 = pow((ior_i - ior_t) / (ior_i + ior_t), 2);
    return R_0 + (1 - R_0) * pow((1 - cos_theta_i), 5);
}

Vector3f PathTracer::directLighting(const Scene& scene, const Ray& ray, const Vector3f& prevHit, const Vector3f& prevN, Vector3f& w_i)
{
    std::vector<Triangle*> emissives = scene.getEmissives();
    Vector3f y_sum(0, 0, 0);
    int sampleNum = 5;

    for (Triangle* emissive : emissives) {
        const tinyobj::material_t& mat = emissive->getMaterial();
        Vector3f y(0, 0, 0);

        for (int rep = 0; rep < sampleNum; rep++) {
            Vector3f samplePoint = sampleOnTriangle(emissive->getVertices());
            w_i = samplePoint - prevHit;
            w_i = w_i.normalized();

            Ray r_i(prevHit, w_i);
            IntersectionInfo i;
            if (!scene.getIntersection(r_i, &i)) {
                continue;
            }

            const Triangle *hitTriangle = static_cast<const Triangle *>(i.data);

            if (hitTriangle->getIndex() != emissive->getIndex()) {
                continue;
            }

            Vector3f l(mat.emission);
            float cos_theta = std::max(w_i.dot(prevN), 0.f);
            float cos_theta_p = std::max(-w_i.dot(hitTriangle->getNormal(i)), 0.f);
            y += l * cos_theta * cos_theta_p / (samplePoint - prevHit).squaredNorm();
        }

        y_sum += getTriangleArea(emissive->getVertices()) * y / sampleNum;
    }

    return y_sum;
}

float PathTracer::getTriangleArea(const Vector3<Eigen::Vector3f>& vertices) {
    Vector3f ab = vertices[1] - vertices[0];
    Vector3f ac = vertices[2] - vertices[0];
    Vector3f cross = ab.cross(ac);
    return 0.5f * cross.norm();
}

Vector3f PathTracer::sampleOnTriangle(const Vector3<Eigen::Vector3f>& vertices) {
    float r1 = dis(gen);
    float r2 = dis(gen);
    float sqrt_r1 = sqrt(r1);

    float u = 1.f - sqrt_r1;
    float v = r2 * sqrt_r1;

    Vector3f p = vertices[0] * u + vertices[1] * v + vertices[2] * (1.f - u - v);
    return p;
}

Vector3f PathTracer::sampleNextDir(Vector3f N)
{
    float phi = 2.f * M_PI * dis(gen); // azimuthal
    float theta = acos(1.f - dis(gen)); // zenith

    // convert to cartesian coordinate
    Vector3f wi(sin(theta) * cos(phi),
                sin(theta) * sin(phi),
                cos(theta));

    // tangent
    Vector3f T;
    float absX = std::fabs(N.x()), absY = std::fabs(N.y()), absZ = std::fabs(N.z());
    if (absX <= absY && absX <= absZ) {
        T = N.cross(Vector3f(1.0f, 0.0f, 0.0f));  // cross with x-axis
    } else if (absY <= absX && absY <= absZ) {
        T = N.cross(Vector3f(0.0f, 1.0f, 0.0f));  // cross with y-axis
    } else {
        T = N.cross(Vector3f(0.0f, 0.0f, 1.0f));  // cross with z-axis
    }
    T.normalize();

    // bitangent
    Vector3f B = N.cross(T);

    // transform matrix
    Eigen::Matrix3f matrix;
    matrix.col(0) = T;
    matrix.col(1) = B;
    matrix.col(2) = N;

    // align surface normal N's local coordinate system
    wi = (matrix * wi).normalized();
    return wi;
}

void PathTracer::toneMap(QRgb *imageData, std::vector<Vector3f> &intensityValues)
{
    for(int y = 0; y < m_height; ++y) {
        for(int x = 0; x < m_width; ++x) {
            int offset = x + (y * m_width);

            // rgb to luminance
            float luminance = intensityValues[offset].dot(Vector3f(0.2126f, 0.7152f, 0.0722f));

            // apply reinhard
            float toneMappedLuminance = luminance / (1.f + luminance);
            float scale = toneMappedLuminance / luminance;
            intensityValues[offset] *= scale;

            // apply gamma correction
            float gamma = 1.f / 2.2f;
            float r = pow(intensityValues[offset][0], gamma);
            float g = pow(intensityValues[offset][1], gamma);
            float b = pow(intensityValues[offset][2], gamma);

            // scale to [0, 255] and clamp
            r = std::min(255.f, std::max(0.f, 255.f * r));
            g = std::min(255.f, std::max(0.f, 255.f * g));
            b = std::min(255.f, std::max(0.f, 255.f * b));

            imageData[offset] = qRgb(r, g, b);
        }
    }
}
