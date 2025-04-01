#include "cuda.h"
#include <CUDAModule.h>
#include <cstdint>
#include <stddef.h>

using namespace driver;

static size_t getImageSize(const void* image) {
    const struct fat_elf_header* header = reinterpret_cast<const struct fat_elf_header*>(image);
    return header->size; 
}

static size_t getPTXSection(uint8_t *decompressed_data, size_t decompressed_size, uint8_t ** ptx_section)
{
    // const struct fat_elf_header* header = reinterpret_cast<const struct fat_elf_header*>(decompressed_data);
    // size_t header_size = header->header_size;
    const struct fat_text_header text_header = *reinterpret_cast<const struct fat_text_header*>(decompressed_data);
    size_t text_header_size = text_header.header_size;
    *ptx_section = decompressed_data + text_header_size;
    return decompressed_size - text_header_size;
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
        LOG(LOG_LEVEL_DEBUG, "DEBUG", "Loading fatbin module");
        uint8_t *decompressedData = nullptr;
        size_t imageSize = getImageSize(image);
        size_t decompressedSize = decompress_fatbin(reinterpret_cast<const uint8_t*>(image), imageSize, &decompressedData);
        // Skip the fatbin text header
        uint8_t *ptx_section = nullptr;
        size_t ptx_size = getPTXSection(decompressedData, decompressedSize, &ptx_section);
        // filter PTX code
        LOG(LOG_LEVEL_DEBUG, "DEBUG", "Filtering PTX code");
        // std::string filtered_ptx;
        // filtered_ptx.reserve(ptx_size);
        // std::copy_if(ptx_section, ptx_section + ptx_size, std::back_inserter(filtered_ptx),
        //     [](unsigned char c) { return (c >= 32) || (c == '\t') || (c == '\n') || (c == '\r'); });
        std::vector<uint8_t> filteredData;
        for (size_t i = 0; i < ptx_size; ++i) {
            uint8_t byte = ptx_section[i];
            if ((byte >= 32) || (byte == '\t') || (byte == '\n') || (byte == '\r')) {
                filteredData.push_back(byte);
            }
        }
        // Load the decompressed ptx data into the module
        LOG(LOG_LEVEL_DEBUG, "DEBUG", "Loading PTX code into module");
        // Create a temporary file to store the PTX code
        std::filesystem::path temp_dir = std::filesystem::temp_directory_path();
        std::filesystem::path temp_file = temp_dir / "translator_temp.ptx";
        std::ofstream ofs(temp_file, std::ios::binary);
        // ofs << filtered_ptx;
        ofs.write(reinterpret_cast<const char*>(filteredData.data()), filteredData.size());
        ofs.close();
        // std::cout << temp_file.string() << std::endl;
        return module.load(temp_file.string());
    }
    // PTX text file
    else {

        std::filesystem::path temp_dir = std::filesystem::temp_directory_path();
        std::filesystem::path temp_file = temp_dir / "translator_temp.ptx";
        std::ofstream ofs(temp_file);
        
        ofs.close();
        return module.load(temp_file.string());
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

std::tuple<CUdeviceptr, size_t> CUDAModule::getGlobal(const char* name) {
    // Check if the global variable is already loaded
    auto it = globals.find(name);
    if (it != globals.end()) {
        auto ptr = it->second->pointer;
        return {static_cast<CUdeviceptr>(reinterpret_cast<uintptr_t>(ptr)), it->second->statement.bytes()};
    }

    auto global = module.getGlobal(name);
    if (global == nullptr) {
        return {0, 0};
    }
    globals.emplace(name, global);
    auto ptr = global->pointer;
    return {static_cast<CUdeviceptr>(reinterpret_cast<uintptr_t>(ptr)), it->second->statement.bytes()};
}