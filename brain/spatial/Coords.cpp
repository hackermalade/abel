#include "Coords.h"
#include <cmath>
#include <stdexcept>
#include <sstream>

namespace abel {

// ═══════════════════════════════════════════════════════════════════════════
//  Coord3D implementation
// ═══════════════════════════════════════════════════════════════════════════
Coord3D::Coord3D(double x_val, double y_val, double z_val)
    : x(x_val), y(y_val), z(z_val) {}

std::string Coord3D::toString() const {
    std::ostringstream oss;
    oss << "(" << x << ", " << y << ", " << z << ")";
    return oss.str();
}

bool Coord3D::operator==(const Coord3D& other) const {
    // Compare with a small epsilon for floating‑point safety
    const double eps = 1e-9;
    return std::abs(x - other.x) < eps &&
           std::abs(y - other.y) < eps &&
           std::abs(z - other.z) < eps;
}

Coord3D Coord3D::operator+(const Coord3D& other) const {
    return Coord3D(x + other.x, y + other.y, z + other.z);
}

Coord3D Coord3D::operator-(const Coord3D& other) const {
    return Coord3D(x - other.x, y - other.y, z - other.z);
}

Coord3D Coord3D::operator*(double scalar) const {
    return Coord3D(x * scalar, y * scalar, z * scalar);
}

Coord3D operator*(double scalar, const Coord3D& c) {
    return c * scalar;
}

Coord3D Coord3D::operator/(double scalar) const {
    if (scalar == 0.0) throw std::domain_error("Division by zero");
    return Coord3D(x / scalar, y / scalar, z / scalar);
}

Coord3D& Coord3D::operator+=(const Coord3D& other) {
    x += other.x; y += other.y; z += other.z;
    return *this;
}

Coord3D& Coord3D::operator-=(const Coord3D& other) {
    x -= other.x; y -= other.y; z -= other.z;
    return *this;
}

Coord3D& Coord3D::operator*=(double scalar) {
    x *= scalar; y *= scalar; z *= scalar;
    return *this;
}

Coord3D& Coord3D::operator/=(double scalar) {
    if (scalar == 0.0) throw std::domain_error("Division by zero");
    x /= scalar; y /= scalar; z /= scalar;
    return *this;
}

double Coord3D::norm() const {
    return std::sqrt(x*x + y*y + z*z);
}

double Coord3D::squaredNorm() const {
    return x*x + y*y + z*z;
}

Coord3D Coord3D::normalised() const {
    double n = norm();
    if (n == 0.0) return Coord3D(0, 0, 0);
    return *this / n;
}

double Coord3D::distanceTo(const Coord3D& other) const {
    return (*this - other).norm();
}

double Coord3D::squaredDistanceTo(const Coord3D& other) const {
    return (*this - other).squaredNorm();
}

double Coord3D::dot(const Coord3D& other) const {
    return x * other.x + y * other.y + z * other.z;
}

Coord3D Coord3D::cross(const Coord3D& other) const {
    return Coord3D(y * other.z - z * other.y,
                   z * other.x - x * other.z,
                   x * other.y - y * other.x);
}

// ── Conduction delay utilities ───────────────────────────────────────────
double Coord3D::conductionDelayStatic(double distance_mm, double speed_mm_per_ms) {
    if (distance_mm < 0.0)
        throw std::invalid_argument("Distance cannot be negative");
    if (speed_mm_per_ms <= 0.0)
        throw std::invalid_argument("Speed must be positive");
    return distance_mm / speed_mm_per_ms;
}

double Coord3D::delayTo(const Coord3D& other, double speed_mm_per_ms) const {
    return conductionDelayStatic(distanceTo(other), speed_mm_per_ms);
}

} // namespace abel
