#include <CUDAModule.h>

using namespace driver;

static size_t getImageSize(const void* image) {
    const struct fat_elf_header* header = reinterpret_cast<const struct fat_elf_header*>(image);
    return header->size; 
}

static size_t getPTXSection(uint8_t *decompressed_data, uint8_t ** ptx_section)
{
    const struct fat_elf_header* header = reinterpret_cast<const struct fat_elf_header*>(decompressed_data);
    size_t header_size = header->header_size;
    const struct fat_text_header text_header = *reinterpret_cast<const struct fat_text_header*>(decompressed_data + header_size);
    size_t text_header_size = text_header.header_size;
    *ptx_section = decompressed_data + header_size + text_header_size;
    return text_header.size;
}

bool CUDAModule::load(const std::string& path) {
    // Load the module from the given path
    return module.load(path);
}

bool CUDAModule::load(const void* image) {
    // check magic number
    const uint32_t magicNumber = *reinterpret_cast<const uint32_t*>(image);
    // ELF magic number (cubin)
    if (magicNumber == 0x464c457f) { 
        return false;
    }
    // fatbin magic number
    else if (magicNumber == 0xba55ed50) { 
        uint8_t *decompressed_data = NULL;
        size_t imageSize = getImageSize(image);
        decompress_fatbin(reinterpret_cast<const uint8_t*>(image), imageSize, &decompressed_data);
        // Load the decompressed data into the module
    }
    // PTX text file
    else {
        
    }
}

void CUDAModule::unload() {
    // Unload the module
    module.unload();
}

ir::Module* CUDAModule::getModule() {
    return &module;
}

CUfunction CUDAModule::getFunction(const char* name) {
    // Check if the function is already loaded
    auto it = functions.find(name);
    if (it != functions.end()) {
        return it->second.getFunctionPointer();
    }

    auto func = module.getKernel(name);
    if (func == nullptr) {
        return nullptr;
    }
    // CUDAFunction cudaFunc(func);
    // functions.emplace(name, CUDAFunction(func)); 
    // return functions.at(name).getFunctionPointer();
    // CUDAFunction cudaFunc(name);
    functions.emplace(name, CUDAFunction(name, this)); 
    return functions.at(name).getFunctionPointer();
}