#include "includes/cnn/ConvolutionLayers.hpp"
#include "cnn/Matrix.hpp"
#include "cnn/ImageInput.hpp"
#include "cnn/Random.hpp"
#include <random>
#include <cmath>

ConvolutionLayers::ConvolutionLayers(gridEntity main_image)
    : raw_image(main_image)
{
    // -------------------------------------------------
    // First convolution layer: fixed feature detectors
    // -------------------------------------------------

    this->predefined_filters.push_back(
        Filters::STRONG_VERTICAL_EDGE_DETECTION
    );

    this->predefined_filters.push_back(
        Filters::STRONG_HORIZONTAL_EDGE_DETECTION
    );

    this->predefined_filters.push_back(
        Filters::STRONG_DIAGONAL_EDGE_DETECTION
    );

    this->no_of_filters_used = 3;
    input_channels.resize(this->no_of_filters_used);


    // -------------------------------------------------
    // Second convolution layer: trainable filters
    // -------------------------------------------------

    this->no_of_filters_in_second_CL = 3;


    // -------------------------------------------------
    // Deterministic random initialization
    // -------------------------------------------------

    auto& gen = Random::engine();

    // Xavier initialization:
    //
    // fan_in  = 3 × 3 × number_of_input_channels
    // fan_out = 3 × 3 × number_of_output_filters

    double fan_in =
        3.0 * 3.0 * this->no_of_filters_used;

    double fan_out =
        3.0 * 3.0 * this->no_of_filters_in_second_CL;

    double std_dev =
        std::sqrt(2.0 / (fan_in + fan_out));

    std::normal_distribution<double>
        distribution(0.0, std_dev);


    // -------------------------------------------------
    // Create trainable filters
    // -------------------------------------------------

    for (int i = 0;
         i < this->no_of_filters_in_second_CL;
         i++)
    {
        volumetricEntity temp;

        for (int j = 0;
             j < this->no_of_filters_used;
             j++)
        {
            gridEntity filter(
                3,
                std::vector<double>(3)
            );

            for (int r = 0; r < 3; r++)
            {
                for (int c = 0; c < 3; c++)
                {
                    filter[r][c] = distribution(gen);
                }
            }

            temp.push_back(filter);
        }

        this->second_layer_filters.push_back(temp);
    }
}

std::vector<gridEntity> ConvolutionLayers::get_all_predefined_filter()
{
    return this->predefined_filters;
}

void ConvolutionLayers::apply_normalaization_universal(gridEntity& to_normailize)
{
    int min = 0;
    int max = 800;

    for (int i = 0; i < to_normailize.size(); i++)
    {
        for (int j = 0; j < to_normailize.at(0).size(); j++)
        {
            to_normailize.at(i).at(j) = (to_normailize.at(i).at(j) - min) / (max - min);
        }
    }
}

gridEntity ConvolutionLayers::apply_filter_universal(gridEntity image, gridEntity filt, int stride)
{
    gridEntity f_map = CNN_Matrix::Matrix::convolute(image, filt, stride);
    return f_map;
}

void ConvolutionLayers::activate_feature_map_using_RELU_universal(gridEntity& to_activate)
{
    for (int i = 0; i < to_activate.size(); i++)
    {
        for (int j = 0; j < to_activate.at(0).size(); j++)
        {
            if (to_activate.at(i).at(j) < 0)
            {
                to_activate.at(i).at(j) = 0;
            }
        }
    }
}

void ConvolutionLayers::activate_feature_map_using_SIGMOID(gridEntity& to_activate)
{
    for (int i = 0; i < to_activate.size(); i++)
    {
        for (int j = 0; j < to_activate.at(0).size(); j++)
        {
            to_activate.at(i).at(j) = (1) / (1 + pow(2.71828, -to_activate.at(i).at(j)));
        }
    }
}

gridEntity ConvolutionLayers::unpool_without_indices(gridEntity& pooled_gradients, gridEntity& original_filter_map,
    int stride, int poolHeight, int poolWidth)
{
    int originalHeight = original_filter_map.size();
    int originalWidth = original_filter_map[0].size();

    gridEntity unpooled_gradients(originalHeight, std::vector<double>(originalWidth, 0.0));

    for (int i = 0; i < pooled_gradients.size(); i++)
    {
        for (int j = 0; j < pooled_gradients[0].size(); j++)
        {
            int startRow = i * stride;
            int startCol = j * stride;

            int endRow = std::min(startRow + poolHeight, originalHeight);
            int endCol = std::min(startCol + poolWidth, originalWidth);

            // Find max position
            double max_val = -1e9;
            int maxRow = startRow, maxCol = startCol;

            for (int row = startRow; row < endRow; row++)
            {
                for (int col = startCol; col < endCol; col++)
                {
                    if (original_filter_map[row][col] > max_val)
                    {
                        max_val = original_filter_map[row][col];
                        maxRow = row;
                        maxCol = col;
                    }
                }
            }

            unpooled_gradients[maxRow][maxCol] += pooled_gradients[i][j];
        }
    }

    return unpooled_gradients;
}

gridEntity ConvolutionLayers::apply_pooling_univeral(gridEntity to_pool, int stride)
{
    int poolHeight = 2;
    int poolWidth = 2;

    int inputHeight = to_pool.size();
    int inputWidth = to_pool[0].size();

    gridEntity pooled;

    for (int i = 0; i < inputHeight; i += stride)
    {
        std::vector<double> pool_row;
        for (int j = 0; j < inputWidth; j += stride)
        {
            double max_val = -1e9;  // Changed from 0 to handle negative values
            for (int kh = i; kh < i + poolHeight; kh++)
            {
                for (int kw = j; kw < j + poolWidth; kw++)
                {
                    if ((kh < inputHeight) && (kw < inputWidth))
                    {
                        max_val = std::max(max_val, to_pool.at(kh).at(kw));
                    }
                }
            }
            pool_row.push_back(max_val);
        }
        pooled.push_back(pool_row);
    }
    return pooled;
}

gridEntity& ConvolutionLayers::get_raw_input_image()
{
    return this->raw_image;
}

std::vector<gridEntity>& ConvolutionLayers::get_feature_map() {
    return this->feature_maps;
}

std::vector<gridEntity>& ConvolutionLayers::get_pool_map() {
    return this->pool_maps;
}

volumetricEntity& ConvolutionLayers::get_input_channels() {
    return this->input_channels;
}

std::vector<volumetricEntity>& ConvolutionLayers::get_all_training_filter()
{
    return this->training_filters;
}

volumetricEntity& ConvolutionLayers::get_output_feature_maps()
{
    return this->ouput_feature_maps;
}

volumetricEntity& ConvolutionLayers::get_final_pool_maps()
{
    return this->final_pool_maps;
}

// Helper function to flip a 2D grid (rotate 180 degrees)
gridEntity flip_grid(const gridEntity& grid) {
    int rows = grid.size();
    int cols = grid[0].size();
    gridEntity flipped(rows, std::vector<double>(cols));

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            flipped[i][j] = grid[rows - 1 - i][cols - 1 - j];
        }
    }
    return flipped;
}


// Add these helper functions to ConvolutionLayers.hpp and ConvolutionLayers.cpp

// Helper: Pad a grid with zeros
gridEntity pad_grid(const gridEntity& input, int padding) {
    int newHeight = input.size() + 2 * padding;
    int newWidth = input[0].size() + 2 * padding;

    gridEntity padded(newHeight, std::vector<double>(newWidth, 0.0));

    for (size_t i = 0; i < input.size(); i++) {
        for (size_t j = 0; j < input[0].size(); j++) {
            padded[i + padding][j + padding] = input[i][j];
        }
    }

    return padded;
}

// CRITICAL FIX: Full convolution for backprop (with padding)
gridEntity ConvolutionLayers::full_convolve(
    const gridEntity& input,
    const gridEntity& filter,
    int stride
) {
    int filterSize = filter.size();
    int padding = filterSize - 1;

    // Flip the filter 180 degrees for proper convolution
    gridEntity flipped_filter(filterSize, std::vector<double>(filterSize));
    for (int i = 0; i < filterSize; i++) {
        for (int j = 0; j < filterSize; j++) {
            flipped_filter[i][j] = filter[filterSize - 1 - i][filterSize - 1 - j];
        }
    }

    // Pad input
    gridEntity padded_input = pad_grid(input, padding);

    int outputHeight = (padded_input.size() - filterSize) / stride + 1;
    int outputWidth = (padded_input[0].size() - filterSize) / stride + 1;

    gridEntity output(outputHeight, std::vector<double>(outputWidth, 0.0));

    for (int i = 0; i < outputHeight; i++) {
        for (int j = 0; j < outputWidth; j++) {
            double sum = 0.0;
            for (int fi = 0; fi < filterSize; fi++) {
                for (int fj = 0; fj < filterSize; fj++) {
                    int inputRow = i * stride + fi;
                    int inputCol = j * stride + fj;
                    sum += padded_input[inputRow][inputCol] * flipped_filter[fi][fj];
                }
            }
            output[i][j] = sum;
        }
    }

    return output;
}

// Compute cross-correlation for filter gradients
gridEntity ConvolutionLayers::cross_correlate(
    const gridEntity& input,
    const gridEntity& gradient,
    int stride
) {
    int filterSize = 3;
    gridEntity filterGrad(filterSize, std::vector<double>(filterSize, 0.0));

    int gradHeight = gradient.size();
    int gradWidth = gradient[0].size();

    for (int fi = 0; fi < filterSize; fi++) {
        for (int fj = 0; fj < filterSize; fj++) {
            double sum = 0.0;

            for (int gi = 0; gi < gradHeight; gi++) {
                for (int gj = 0; gj < gradWidth; gj++) {
                    int inputRow = gi * stride + fi;
                    int inputCol = gj * stride + fj;

                    if (inputRow < input.size() && inputCol < input[0].size()) {
                        sum += input[inputRow][inputCol] * gradient[gi][gj];
                    }
                }
            }

            filterGrad[fi][fj] = sum;
        }
    }

    return filterGrad;
}

// Apply ReLU derivative
void ConvolutionLayers::apply_relu_derivative(
    gridEntity& gradients,
    const gridEntity& original_output
) {
    for (size_t i = 0; i < gradients.size(); i++) {
        for (size_t j = 0; j < gradients[0].size(); j++) {
            if (original_output[i][j] <= 0) {
                gradients[i][j] = 0.0;
            }
        }
    }
}

// Compute filter gradients
std::vector<volumetricEntity> ConvolutionLayers::compute_filter_gradients(
    const std::vector<gridEntity>& inputChannels,
    const std::vector<gridEntity>& outputGradients,
    int stride
) {
    std::vector<volumetricEntity> filterGradients;

    int numFilters = outputGradients.size();
    int numChannels = inputChannels.size();

    for (int filterIdx = 0; filterIdx < numFilters; ++filterIdx) {
        volumetricEntity filterGradient(numChannels);

        for (int channelIdx = 0; channelIdx < numChannels; ++channelIdx) {
            gridEntity gradient = cross_correlate(
                inputChannels[channelIdx],
                outputGradients[filterIdx],
                stride
            );

            filterGradient[channelIdx] = gradient;
        }

        filterGradients.push_back(filterGradient);
    }

    return filterGradients;
}

// CRITICAL: Compute input gradients (for backprop to previous layer)
std::vector<gridEntity> ConvolutionLayers::compute_input_gradients(
    const std::vector<gridEntity>& outputGradients,
    const std::vector<volumetricEntity>& filters,
    int stride
) {
    int numChannels = filters[0].size(); // Number of input channels
    int numFilters = filters.size();     // Number of output channels

    // Initialize input gradients
    std::vector<gridEntity> inputGradients;
    for (int c = 0; c < numChannels; c++) {
        int height = outputGradients[0].size() + 2; // Approximate size
        int width = outputGradients[0][0].size() + 2;
        inputGradients.push_back(gridEntity(height, std::vector<double>(width, 0.0)));
    }

    // For each output gradient
    for (int f = 0; f < numFilters; f++) {
        // For each input channel
        for (int c = 0; c < numChannels; c++) {
            // Convolve gradient with filter
            gridEntity channelGrad = full_convolve(
                outputGradients[f],
                filters[f][c],
                stride
            );

            // Accumulate into input gradients
            for (size_t i = 0; i < channelGrad.size() && i < inputGradients[c].size(); i++) {
                for (size_t j = 0; j < channelGrad[0].size() && j < inputGradients[c][0].size(); j++) {
                    inputGradients[c][i][j] += channelGrad[i][j];
                }
            }
        }
    }

    return inputGradients;
}

// Update filters with gradient clipping and optional momentum
void ConvolutionLayers::update_filters_with_gradients(
    std::vector<volumetricEntity>& filters,
    const std::vector<volumetricEntity>& gradients,
    double learningRate
)
{
    const double GRAD_CLIP = 1.0; // Lower clip threshold

    for (size_t f = 0; f < filters.size() && f < gradients.size(); ++f)
    {
        for (size_t c = 0; c < filters[f].size() && c < gradients[f].size(); ++c)
        {
            if (filters[f][c].size() != gradients[f][c].size() ||
                filters[f][c][0].size() != gradients[f][c][0].size())
            {
                std::cerr << "Dimension mismatch at filter " << f << ", channel " << c << "\n";
                std::cerr << "Filter: " << filters[f][c].size() << "x" << filters[f][c][0].size()
                    << ", Grad: " << gradients[f][c].size() << "x" << gradients[f][c][0].size() << "\n";
                continue;
            }

            for (size_t i = 0; i < filters[f][c].size(); ++i)
            {
                for (size_t j = 0; j < filters[f][c][0].size(); ++j)
                {
                    double grad = gradients[f][c][i][j];

                    // Clip gradient
                    if (grad > GRAD_CLIP) grad = GRAD_CLIP;
                    if (grad < -GRAD_CLIP) grad = -GRAD_CLIP;

                    // Update weight
                    filters[f][c][i][j] -= learningRate * grad;
                }
            }
        }
    }
}