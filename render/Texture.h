#pragma once
#include <vector>
#include <string>

namespace abel {

class Texture {
public:
    Texture();
    ~Texture();

    Texture(Texture&& other) noexcept;
    Texture& operator=(Texture&& other) noexcept;

    void create(int width, int height, int channels);
    bool loadFromFile(const std::string& path);
    bool saveToFile(const std::string& path) const;

    void setPixel(int x, int y,
                  unsigned char r, unsigned char g,
                  unsigned char b, unsigned char a = 255);
    void getPixel(int x, int y,
                  unsigned char& r, unsigned char& g,
                  unsigned char& b, unsigned char& a) const;

    void setFiltering(bool linear = true);
    void generateMipmaps();

    int width() const;
    int height() const;
    int channels() const;
    bool hasMipmaps() const;
    bool isValid() const;
    const unsigned char* data() const;
    size_t dataSize() const;

private:
    int width_, height_, channels_;
    std::vector<unsigned char> data_;
    bool filtering_;
    bool mipmaps_;
    bool valid_;
};

} // namespace abel
