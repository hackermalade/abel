#pragma once
#include <unordered_map>
#include "../brain/spatial/Coords.h"
#include <glm/glm.hpp>

struct GLFWwindow;

namespace abel {

class World;
class Camera;
class Mesh;
struct MeshGPU;

class Renderer {
public:
    Renderer();
    ~Renderer();

    bool init();
    void shutdown();
    void draw(const World& world, const Camera& camera);

private:
    MeshGPU& getOrCreateGPU(const Mesh& mesh);

    GLFWwindow* window_;
    unsigned int pbrProgram_;
    int loc_uModel_, loc_uViewProj_, loc_uNormalMat_, loc_uCamPos_;
    int loc_uLightDir_, loc_uLightColor_, loc_uAmbient_;
    int loc_uAlbedoTex_, loc_uHasTexture_;

    glm::vec3 lightDir_;
    glm::vec3 lightColor_;
    float ambient_;

    std::unordered_map<const Mesh*, MeshGPU> gpuCache_;
};

} // namespace abel
