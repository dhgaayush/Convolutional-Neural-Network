#include "cnn/ImageInput.hpp"
#include "cnn/Matrix.hpp"
#include "cnn/ConvolutionLayers.hpp"
#include "cnn/DeepNetwork.hpp"
#include "cnn/ModelSerializer.hpp"
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <iostream>
#include <algorithm>

// =========================================================================
// Constants
// =========================================================================
static const std::string DEFAULT_MODEL_PATH = "saved_model.bin";
static const int         NUM_CLASSES = 10;
static const int         IMAGE_SIZE = 28;

// =========================================================================
// MNIST helpers
// =========================================================================
struct MNISTSample {
    int label;
    std::vector<std::vector<double>> pixels;
};

std::vector<MNISTSample> loadMNISTFromCSV(const std::string& filepath, int maxSamples = -1) {
    std::vector<MNISTSample> dataset;
    std::ifstream file(filepath);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filepath << std::endl;
        return dataset;
    }
    std::string line;
    std::getline(file, line); // skip header
    int count = 0;
    while (std::getline(file, line) && (maxSamples == -1 || count < maxSamples)) {
        std::stringstream ss(line);
        std::string value;
        MNISTSample sample;
        std::getline(ss, value, ',');
        sample.label = std::stoi(value);
        std::vector<double> flat;
        while (std::getline(ss, value, ','))
            flat.push_back(std::stod(value) / 255.0);
        sample.pixels.resize(IMAGE_SIZE, std::vector<double>(IMAGE_SIZE));
        for (int i = 0; i < IMAGE_SIZE; i++)
            for (int j = 0; j < IMAGE_SIZE; j++)
                sample.pixels[i][j] = flat[i * IMAGE_SIZE + j];
        dataset.push_back(sample);
        count++;
        if (count % 1000 == 0)
            std::cout << "Loaded " << count << " samples..." << std::endl;
    }
    std::cout << "Total loaded: " << dataset.size() << std::endl;
    return dataset;
}

std::vector<double> createOneHot(int label) {
    std::vector<double> t(NUM_CLASSES, 0.0);
    t[label] = 1.0;
    return t;
}

// =========================================================================
// Conv pipeline -- shared between train, eval, predict
// =========================================================================
std::vector<double> processImageThroughConvLayers(
    ConvolutionLayers& conv,
    const std::vector<std::vector<double>>& pixels
) {
    conv.get_feature_map().clear();
    conv.get_pool_map().clear();
    conv.get_output_feature_maps().clear();
    conv.get_final_pool_maps().clear();
    conv.get_raw_input_image() = pixels;

    // Layer 1: hardcoded edge filters
    for (const auto& filter : conv.get_all_predefined_filter()) {
        gridEntity fm = conv.apply_filter_universal(conv.get_raw_input_image(), filter, 1);
        conv.activate_feature_map_using_RELU_universal(fm);
        conv.get_feature_map().push_back(fm);
    }
    for (const auto& fm : conv.get_feature_map())
        conv.get_pool_map().push_back(conv.apply_pooling_univeral(fm, 2));

    conv.get_input_channels() = conv.get_pool_map();

    // Layer 2: trainable filters
    std::vector<gridEntity> layer2Maps;
    for (const auto& trainingFilter : conv.get_all_training_filter()) {
        std::vector<gridEntity> perChannel;
        for (size_t ci = 0; ci < conv.get_input_channels().size(); ci++)
            perChannel.push_back(conv.apply_filter_universal(conv.get_input_channels()[ci], trainingFilter[ci], 1));
        gridEntity summed = CNN_Matrix::Matrix::sum_of_all_matrix_elements(perChannel);
        conv.activate_feature_map_using_RELU_universal(summed);
        layer2Maps.push_back(summed);
    }
    conv.get_output_feature_maps() = layer2Maps;

    for (const auto& fm : conv.get_output_feature_maps())
        conv.get_final_pool_maps().push_back(conv.apply_pooling_univeral(fm, 2));

    // Flatten
    std::vector<double> flat;
    for (const auto& pm : conv.get_final_pool_maps())
        for (const auto& row : pm)
            flat.insert(flat.end(), row.begin(), row.end());
    return flat;
}

// =========================================================================
// TRAIN MODE
// =========================================================================
void runTraining(const std::string& modelPath) {
    std::cout << "Loading MNIST training data..." << std::endl;
    std::vector<MNISTSample> trainData = loadMNISTFromCSV("resource/mnist_train.csv", 5000);
    if (trainData.empty()) { std::cerr << "No training data. Exiting.\n"; return; }

    std::vector<std::vector<double>> dummy(IMAGE_SIZE, std::vector<double>(IMAGE_SIZE, 0.0));
    ConvolutionLayers conv(dummy);

    std::vector<double> sample = processImageThroughConvLayers(conv, trainData[0].pixels);
    int inputSize = (int)sample.size();
    std::cout << "Flattened feature vector size: " << inputSize << std::endl;

    const double nnLR = 0.001;
    const double convLR = 0.00001;
    std::vector<int> topology = { inputSize, 256, NUM_CLASSES };
    DeepNetwork net(topology, nnLR);

    const int EPOCHS = 15;
    const int outputIdx = (int)topology.size() - 1;
    double currentConvLR = convLR;

    for (int epoch = 0; epoch < EPOCHS; epoch++) {
        if (epoch > 0 && epoch % 5 == 0) {
            currentConvLR *= 0.5;
            std::cout << "  [LR Decay] Conv=" << currentConvLR << std::endl;
        }

        double epochError = 0.0;
        int    correct = 0;
        std::cout << "\n==============================\n";
        std::cout << "Epoch " << (epoch + 1) << "/" << EPOCHS << "\n";
        std::cout << "==============================" << std::endl;

        for (size_t i = 0; i < trainData.size(); i++) {
            MNISTSample& s = trainData[i];
            std::vector<double> target = createOneHot(s.label);

            // Forward
            std::vector<double> flat = processImageThroughConvLayers(conv, s.pixels);
            net.setCurrentInput(flat);
            net.setTarget(target);
            net.forwardPropogation();
            net.setErrors();
            epochError += net.getGlobalError();

            // Accuracy check
            int predLabel = 0; double maxP = 0.0;
            for (int j = 0; j < NUM_CLASSES; j++) {
                double p = net.GetLayer(outputIdx)->getNeuron(j)->getActivatedVal();
                if (p > maxP) { maxP = p; predLabel = j; }
            }
            if (predLabel == s.label) correct++;

            // Backward -- FC
            net.gardientComputation();
            auto gradMats = net.GetGradientMatrices();
            net.updateWeights();

            // Backward -- Conv layer 2
            int nCh = (int)conv.get_final_pool_maps().size();
            int gH = (int)conv.get_final_pool_maps()[0].size();
            int gW = (int)conv.get_final_pool_maps()[0][0].size();

            GeneralMatrix::Matrix* lastGrad = gradMats.back();
            GeneralMatrix::Matrix* transposedW = net.GetWeightMatrices()[0]->tranpose();
            GeneralMatrix::Matrix* reshapedGrads = *lastGrad * transposedW;

            volumetricEntity pooledGradients;
            int col = 0;
            for (int ch = 0; ch < nCh; ch++) {
                gridEntity chGrad;
                for (int r = 0; r < gH; r++) {
                    std::vector<double> rowG;
                    for (int c = 0; c < gW; c++)
                        rowG.push_back(reshapedGrads->getVal(0, col++));
                    chGrad.push_back(rowG);
                }
                pooledGradients.push_back(chGrad);
            }

            volumetricEntity unpooled;
            for (int ch = 0; ch < nCh; ch++)
                unpooled.push_back(conv.unpool_without_indices(
                    pooledGradients[ch], conv.get_output_feature_maps()[ch], 2, 2, 2));

            for (int ch = 0; ch < nCh; ch++)
                conv.apply_relu_derivative(unpooled[ch], conv.get_output_feature_maps()[ch]);

            auto filterGrads = conv.compute_filter_gradients(conv.get_input_channels(), unpooled, 1);
            conv.update_filters_with_gradients(conv.get_all_training_filter(), filterGrads, currentConvLR);

            delete transposedW;
            delete reshapedGrads;

            if ((i + 1) % 100 == 0)
                std::cout << "  [" << (i + 1) << "/" << trainData.size() << "]"
                << "  AvgLoss=" << (epochError / (i + 1))
                << "  Acc=" << (100.0 * correct / (i + 1)) << "%" << std::endl;

        }

        std::cout << "\n  Epoch Summary | Loss=" << (epochError / trainData.size())
            << " | Acc=" << (100.0 * correct / trainData.size()) << "%"
            << " | " << correct << "/" << trainData.size() << " correct" << std::endl;

        std::cout << "Weight[0][0] going into epoch " << (epoch + 2) << ": "
            << net.GetWeightMatrices()[0]->getVal(0, 0) << std::endl;

        // Checkpoint after every epoch
        std::string ckpt = "checkpoint_epoch_" + std::to_string(epoch + 1) + ".bin";
        ModelSerializer::save(ckpt, conv, net);
    }

    ModelSerializer::save(modelPath, conv, net);
    std::cout << "\nTraining complete. Model saved to: " << modelPath << std::endl;
}

// =========================================================================
// EVALUATE MODE
// =========================================================================
void runEvaluation(const std::string& modelPath) {
    std::cout << "Loading MNIST test data..." << std::endl;
    std::vector<MNISTSample> testData = loadMNISTFromCSV("resource/mnist_test.csv", 1000);
    if (testData.empty()) { std::cerr << "No test data. Exiting.\n"; return; }

    std::vector<std::vector<double>> dummy(IMAGE_SIZE, std::vector<double>(IMAGE_SIZE, 0.0));
    ConvolutionLayers conv(dummy);

    std::vector<double> s = processImageThroughConvLayers(conv, testData[0].pixels);
    int inputSize = (int)s.size();

    std::vector<int> topology = { inputSize, 256, NUM_CLASSES };
    DeepNetwork net(topology, 0.0);
    ModelSerializer::load(modelPath, conv, net);

    int correct = 0;
    int outputIdx = (int)topology.size() - 1;

    for (size_t i = 0; i < testData.size(); i++) {
        std::vector<double> flat = processImageThroughConvLayers(conv, testData[i].pixels);
        net.setCurrentInput(flat);
        net.forwardPropogation();

        int predLabel = 0; double maxP = 0.0;
        for (int j = 0; j < NUM_CLASSES; j++) {
            double p = net.GetLayer(outputIdx)->getNeuron(j)->getActivatedVal();
            if (p > maxP) { maxP = p; predLabel = j; }
        }
        if (predLabel == testData[i].label) correct++;

        if ((i + 1) % 100 == 0)
            std::cout << "Evaluated " << (i + 1) << "/" << testData.size()
            << " | Acc=" << (100.0 * correct / (i + 1)) << "%" << std::endl;
    }

    std::cout << "\nFinal Test Accuracy: "
        << (100.0 * correct / testData.size()) << "%"
        << " (" << correct << "/" << testData.size() << ")" << std::endl;
}

// =========================================================================
// PREDICT MODE -- single image inference
// =========================================================================
void runPredict(const std::string& modelPath, const std::string& imagePath) {
    ImageInput img(imagePath, 0); // 0 = grayscale
    std::vector<std::vector<double>> pixels = img.getMatrixifiedPixelValues();
    if (pixels.empty()) { std::cerr << "Could not load image: " << imagePath << "\n"; return; }

    std::vector<std::vector<double>> dummy(IMAGE_SIZE, std::vector<double>(IMAGE_SIZE, 0.0));
    ConvolutionLayers conv(dummy);
    std::vector<double> s = processImageThroughConvLayers(conv, dummy);
    int inputSize = (int)s.size();

    std::vector<int> topology = { inputSize, 256, NUM_CLASSES };
    DeepNetwork net(topology, 0.0);
    ModelSerializer::load(modelPath, conv, net);

    std::vector<double> flat = processImageThroughConvLayers(conv, pixels);
    net.setCurrentInput(flat);
    net.forwardPropogation();

    int outputIdx = (int)topology.size() - 1;
    std::cout << "\nClass probabilities:" << std::endl;
    int predLabel = 0; double maxP = 0.0;
    for (int j = 0; j < NUM_CLASSES; j++) {
        double p = net.GetLayer(outputIdx)->getNeuron(j)->getActivatedVal();
        std::cout << "  " << j << ": " << (p * 100.0) << "%" << std::endl;
        if (p > maxP) { maxP = p; predLabel = j; }
    }
    std::cout << "\nPredicted digit: " << predLabel
        << "  (confidence: " << (maxP * 100.0) << "%)" << std::endl;
}

// =========================================================================
// Entry point
//
//   ./cnn train   [model.bin]
//   ./cnn eval    [model.bin]
//   ./cnn predict <image.png> [model.bin]
// =========================================================================
int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cout << "Usage:\n"
            << "  " << argv[0] << " train   [model_path]          -- train and save\n"
            << "  " << argv[0] << " eval    [model_path]          -- evaluate on test set\n"
            << "  " << argv[0] << " predict <image.png> [model]   -- predict a single image\n";
        return 1;
    }

    std::string mode = argv[1];
    std::string modelPath = DEFAULT_MODEL_PATH;

    if (mode == "train") {
        if (argc >= 3) modelPath = argv[2];
        runTraining(modelPath);

    }
    else if (mode == "eval") {
        if (argc >= 3) modelPath = argv[2];
        runEvaluation(modelPath);

    }
    else if (mode == "predict") {
        if (argc < 3) { std::cerr << "predict mode requires an image path.\n"; return 1; }
        std::string imagePath = argv[2];
        if (argc >= 4) modelPath = argv[3];
        runPredict(modelPath, imagePath);

    }
    else {
        std::cerr << "Unknown mode: " << mode << std::endl;
        return 1;
    }

    return 0;
}