#pragma once
#include "all_includes.hpp"
#include "DeepNetwork.hpp"
#include "ConvolutionLayers.hpp"
#include <fstream>
#include <string>
#include <stdexcept>

// ============================================================
// ModelSerializer
// Saves and loads all trained weights to/from a binary file.
//
// File format:
//   [HEADER]
//   Magic number (uint32)          - sanity check
//   Version     (uint32)
//
//   [CONV FILTERS SECTION]
//   num_training_filters (int)     - e.g. 3
//   num_channels_per_filter (int)  - e.g. 3
//   filter_height (int)            - e.g. 3
//   filter_width  (int)            - e.g. 3
//   raw filter values (doubles)
//
//   [FC NETWORK SECTION]
//   num_weight_matrices (int)
//   for each weight matrix:
//     rows (int), cols (int)
//     raw values (doubles)
//   num_bias_matrices (int)
//   for each bias matrix:
//     rows (int), cols (int)
//     raw values (doubles)
// ============================================================

class ModelSerializer
{
    static constexpr uint32_t MAGIC = 0xC0010001;
    static constexpr uint32_t VERSION = 1;

public:

    // ----------------------------------------------------------
    // Save
    // ----------------------------------------------------------
    static void save(
        const std::string& filepath,
        ConvolutionLayers& conv,
        DeepNetwork& net
    ) {
        std::ofstream f(filepath, std::ios::binary);
        if (!f.is_open())
            throw std::runtime_error("ModelSerializer::save -- cannot open file: " + filepath);

        // Header
        write<uint32_t>(f, MAGIC);
        write<uint32_t>(f, VERSION);

        // ----- Conv filters -----
        const auto& filters = conv.get_all_training_filter();   // vector<volumetricEntity>
        int nFilters = (int)filters.size();
        int nChannels = nFilters > 0 ? (int)filters[0].size() : 0;
        int fH = (nFilters > 0 && nChannels > 0) ? (int)filters[0][0].size() : 0;
        int fW = (fH > 0) ? (int)filters[0][0][0].size() : 0;

        write<int>(f, nFilters);
        write<int>(f, nChannels);
        write<int>(f, fH);
        write<int>(f, fW);

        for (int fi = 0; fi < nFilters; fi++)
            for (int ci = 0; ci < nChannels; ci++)
                for (int r = 0; r < fH; r++)
                    for (int c = 0; c < fW; c++)
                        write<double>(f, filters[fi][ci][r][c]);

        // ----- FC weight matrices -----
        auto weightMats = net.GetWeightMatrices();
        write<int>(f, (int)weightMats.size());
        for (auto* m : weightMats) {
            write<int>(f, m->getNumRow());
            write<int>(f, m->getNumCols());
            for (int r = 0; r < m->getNumRow(); r++)
                for (int c = 0; c < m->getNumCols(); c++)
                    write<double>(f, m->getVal(r, c));
        }

        // ----- FC bias matrices -----
        auto biasMats = net.GetBiasMatrices();
        write<int>(f, (int)biasMats.size());
        for (auto* m : biasMats) {
            write<int>(f, m->getNumRow());
            write<int>(f, m->getNumCols());
            for (int r = 0; r < m->getNumRow(); r++)
                for (int c = 0; c < m->getNumCols(); c++)
                    write<double>(f, m->getVal(r, c));
        }

        f.close();
        std::cout << "[ModelSerializer] Model saved to: " << filepath << std::endl;
    }

    // ----------------------------------------------------------
    // Load
    // ----------------------------------------------------------
    static void load(
        const std::string& filepath,
        ConvolutionLayers& conv,
        DeepNetwork& net
    ) {
        std::ifstream f(filepath, std::ios::binary);
        if (!f.is_open())
            throw std::runtime_error("ModelSerializer::load -- cannot open file: " + filepath);

        // Header
        uint32_t magic = read<uint32_t>(f);
        uint32_t version = read<uint32_t>(f);

        if (magic != MAGIC)
            throw std::runtime_error("ModelSerializer::load -- invalid file (bad magic number)");
        if (version != VERSION)
            throw std::runtime_error("ModelSerializer::load -- unsupported version: " + std::to_string(version));

        // ----- Conv filters -----
        int nFilters = read<int>(f);
        int nChannels = read<int>(f);
        int fH = read<int>(f);
        int fW = read<int>(f);

        auto& filters = conv.get_all_training_filter();
        filters.clear();
        filters.resize(nFilters);

        for (int fi = 0; fi < nFilters; fi++) {
            filters[fi].resize(nChannels);
            for (int ci = 0; ci < nChannels; ci++) {
                filters[fi][ci].resize(fH, std::vector<double>(fW, 0.0));
                for (int r = 0; r < fH; r++)
                    for (int c = 0; c < fW; c++)
                        filters[fi][ci][r][c] = read<double>(f);
            }
        }

        // ----- FC weight matrices -----
        int nWeightMats = read<int>(f);
        auto weightMats = net.GetWeightMatrices();

        if ((int)weightMats.size() != nWeightMats)
            throw std::runtime_error("ModelSerializer::load -- weight matrix count mismatch");

        for (int mi = 0; mi < nWeightMats; mi++) {
            int rows = read<int>(f);
            int cols = read<int>(f);
            for (int r = 0; r < rows; r++)
                for (int c = 0; c < cols; c++)
                    weightMats[mi]->setVal(r, c, read<double>(f));
        }

        // ----- FC bias matrices -----
        int nBiasMats = read<int>(f);
        auto biasMats = net.GetBiasMatrices();

        if ((int)biasMats.size() != nBiasMats)
            throw std::runtime_error("ModelSerializer::load -- bias matrix count mismatch");

        for (int mi = 0; mi < nBiasMats; mi++) {
            int rows = read<int>(f);
            int cols = read<int>(f);
            for (int r = 0; r < rows; r++)
                for (int c = 0; c < cols; c++)
                    biasMats[mi]->setVal(r, c, read<double>(f));
        }

        f.close();
        std::cout << "[ModelSerializer] Model loaded from: " << filepath << std::endl;
    }

private:
    template<typename T>
    static void write(std::ofstream& f, T val) {
        f.write(reinterpret_cast<const char*>(&val), static_cast<std::streamsize>(sizeof(T)));
    }

    template<typename T>
    static T read(std::ifstream& f) {
        T val{};
        f.read(reinterpret_cast<char*>(&val), static_cast<std::streamsize>(sizeof(T)));
        return val;
    }
};