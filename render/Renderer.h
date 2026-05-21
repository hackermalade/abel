#pragma once

#include <unordered_map>
#include <glm/glm.hpp>

// Forward declarations
struct GLFWwindow;
namespace abel {
class World;
class Camera;
class Mesh;
struct MeshGPU;
}

namespace abel {

class Renderer {
public:
    Renderer();
    ~Renderer();

    bool init();
    void shutdown();
    void draw(const World& world, const Camera& camera);
    GLFWwindow* getWindow() const { return window_; }

private:
    MeshGPU& getOrCreateGPU(const Mesh& mesh);

    GLFWwindow* window_ = nullptr;
    unsigned int pbrProgram_ = 0;
    int loc_uModel_       = -1;
    int loc_uViewProj_    = -1;
    int loc_uNormalMat_   = -1;
    int loc_uCamPos_      = -1;
    int loc_uLightDir_    = -1;
    int loc_uLightColor_  = -1;
    int loc_uAmbient_     = -1;
    int loc_uAlbedoTex_   = -1;
    int loc_uHasTexture_  = -1;

    glm::vec3 lightDir_   = glm::vec3(-0.2f, -0.8f, -0.5f);
    glm::vec3 lightColor_ = glm::vec3(1.0f, 0.95f, 0.85f);
    float ambient_        = 0.15f;

    std::unordered_map<const Mesh*, MeshGPU> gpuCache_;
};

} // namespace abel
