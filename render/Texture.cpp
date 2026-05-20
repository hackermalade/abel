#include "Texture.h"
#include <stb_image.h>
#include <stb_image_write.h>

#include <stdexcept>
#include <cstring>
#include <algorithm>
#include <iostream>

namespace abel {

// ── Default constructor ──────────────────────────────────────────────────
Texture::Texture()
    : width_(0), height_(0), channels_(0),
      filtering_(true), mipmaps_(false), valid_(false)
{}

// ── Destructor ───────────────────────────────────────────────────────────
Texture::~Texture() {}

// ── Move constructor / assignment ────────────────────────────────────────
Texture::Texture(Texture&& other) noexcept
    : width_(other.width_), height_(other.height_), channels_(other.channels_),
      data_(std::move(other.data_)),
      filtering_(other.filtering_), mipmaps_(other.mipmaps_), valid_(other.valid_)
{
    other.valid_ = false;
}

Texture& Texture::operator=(Texture&& other) noexcept {
    if (this != &other) {
        width_ = other.width_;
        height_ = other.height_;
        channels_ = other.channels_;
        data_ = std::move(other.data_);
        filtering_ = other.filtering_;
        mipmaps_ = other.mipmaps_;
        valid_ = other.valid_;
        other.valid_ = false;
    }
    return *this;
}

// ── Create an empty texture with specified dimensions ────────────────────
void Texture::create(int width, int height, int channels) {
    if (width <= 0 || height <= 0 || channels <= 0 || channels > 4)
        throw std::invalid_argument("Invalid texture dimensions or channels");

    width_ = width;
    height_ = height;
    channels_ = channels;
    data_.resize(width * height * channels, 0);
    valid_ = true;
    filtering_ = true;
    mipmaps_ = false;
}

// ── Load texture from an image file (PNG, JPG, BMP, etc.) ─────────────────
bool Texture::loadFromFile(const std::string& path) {
    // Free any existing data
    data_.clear();
    valid_ = false;

    int w, h, comp;
    unsigned char* raw = stbi_load(path.c_str(), &w, &h, &comp, 0);
    if (!raw) {
        std::cerr << "Texture load failed: " << path << " (" 
                  << stbi_failure_reason() << ")" << std::endl;
        return false;
    }

    width_ = w;
    height_ = h;
    channels_ = comp;
    data_.resize(w * h * comp);
    std::memcpy(data_.data(), raw, w * h * comp);
    stbi_image_free(raw);
    valid_ = true;
    return true;
}

// ── Save texture to a PNG file ───────────────────────────────────────────
bool Texture::saveToFile(const std::string& path) const {
    if (!valid_) return false;
    int result = stbi_write_png(path.c_str(), width_, height_, channels_,
                                data_.data(), width_ * channels_);
    return result != 0;
}

// ── Set a single pixel ───────────────────────────────────────────────────
void Texture::setPixel(int x, int y,
                       unsigned char r, unsigned char g,
                       unsigned char b, unsigned char a) {
    if (!valid_ || x < 0 || x >= width_ || y < 0 || y >= height_)
        return;

    size_t idx = (static_cast<size_t>(y) * width_ + x) * channels_;
    data_[idx] = r;
    if (channels_ > 1) data_[idx + 1] = g;
    if (channels_ > 2) data_[idx + 2] = b;
    if (channels_ > 3) data_[idx + 3] = a;
}

// ── Get pixel (read‑only) ────────────────────────────────────────────────
void Texture::getPixel(int x, int y,
                       unsigned char& r, unsigned char& g,
                       unsigned char& b, unsigned char& a) const {
    if (!valid_ || x < 0 || x >= width_ || y < 0 || y >= height_) {
        r = g = b = a = 0;
        return;
    }
    size_t idx = (static_cast<size_t>(y) * width_ + x) * channels_;
    r = data_[idx];
    g = (channels_ > 1) ? data_[idx + 1] : 0;
    b = (channels_ > 2) ? data_[idx + 2] : 0;
    a = (channels_ > 3) ? data_[idx + 3] : 255;
}

// ── Set filtering mode ───────────────────────────────────────────────────
void Texture::setFiltering(bool linear) {
    filtering_ = linear;
}

// ── Flag mipmaps (actual GPU generation in renderer) ─────────────────────
void Texture::generateMipmaps() {
    mipmaps_ = true;
}

// ── Accessors ────────────────────────────────────────────────────────────
int Texture::width() const { return width_; }
int Texture::height() const { return height_; }
int Texture::channels() const { return channels_; }
bool Texture::hasMipmaps() const { return mipmaps_; }
bool Texture::isValid() const { return valid_; }
const unsigned char* Texture::data() const { return data_.data(); }
size_t Texture::dataSize() const { return data_.size(); }

} // namespace abel
