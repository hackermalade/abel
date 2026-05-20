#pragma once

#include <string>
#include <stdexcept>
#include <cmath>
#include <sstream>

namespace abel {

class Coord3D {
public:
    double x, y, z;

    Coord3D(double x_val = 0.0, double y_val = 0.0, double z_val = 0.0);

    std::string toString() const;

    bool operator==(const Coord3D& other) const;
    bool operator!=(const Coord3D& other) const { return !(*this == other); }

    Coord3D operator+(const Coord3D& other) const;
    Coord3D operator-(const Coord3D& other) const;
    Coord3D operator*(double scalar) const;
    friend Coord3D operator*(double scalar, const Coord3D& c);
    Coord3D operator/(double scalar) const;

    Coord3D& operator+=(const Coord3D& other);
    Coord3D& operator-=(const Coord3D& other);
    Coord3D& operator*=(double scalar);
    Coord3D& operator/=(double scalar);

    double norm() const;
    double squaredNorm() const;
    Coord3D normalised() const;

    double distanceTo(const Coord3D& other) const;
    double squaredDistanceTo(const Coord3D& other) const;

    double dot(const Coord3D& other) const;
    Coord3D cross(const Coord3D& other) const;

    // Conduction delay utilities
    static double conductionDelayStatic(double distance_mm, double speed_mm_per_ms = 3.0);
    double delayTo(const Coord3D& other, double speed_mm_per_ms = 3.0) const;
};

} // namespace abel
