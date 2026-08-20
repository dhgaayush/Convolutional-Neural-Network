#include "cnn/Layer.hpp"
#include <assert.h>
#include <cmath>
#include <algorithm>
#include <fstream>
#include "cnn/DeepNetwork.hpp"

using namespace std;

DeepNetwork::DeepNetwork(vector<int> topology, double lr)
{
    this->learningRate = lr;
    this->topology = topology;
    this->topologySize = topology.size();

    for (int i = 0; i < topologySize; i++)
    {
        if (i == topologySize - 1)
        {
            Layer* l = new Layer(topology[i], true);
            this->layers.push_back(l);
        }
        else {
            Layer* l = new Layer(topology[i]);
            this->layers.push_back(l);
        }
    }

    for (int i = 0; i < topologySize - 1; i++)
    {
        GeneralMatrix::Matrix* mw = new GeneralMatrix::Matrix(topology[i], topology[i + 1], true);
        this->weightMatrices.push_back(mw);

        GeneralMatrix::Matrix* mb = new GeneralMatrix::Matrix(1, topology[i + 1], true);
        this->BaisMatrices.push_back(mb);
    }

    histErrors.push_back(1);
}

double DeepNetwork::lastEpoachError()
{
    return histErrors[histErrors.size() - 1];
}

void DeepNetwork::printHistErrors()
{
    cout << "\n Printing Errors from all epochs:" << endl;
    for (int i = 0; i < this->histErrors.size(); i++)
    {
        cout << "Epoch: " << i << ", Error:" << histErrors.at(i) << " \n ";
    }
}

void DeepNetwork::saveHistErrors()
{
    ofstream outFile("error_vs_epoch.csv");
    if (outFile.is_open())
    {
        outFile << "Epoch,Error\n";
        for (size_t i = 0; i < this->histErrors.size(); ++i)
        {
            outFile << i << "," << histErrors[i] << "\n";
        }
        outFile.close();
    }
    else
    {
        cerr << "Unable to open file for writing.\n";
    }
}

vector<double> DeepNetwork::gethisterrors()
{
    return histErrors;
}

double DeepNetwork::getLearningRate()
{
    return learningRate;
}

void DeepNetwork::setTarget(vector<double> target)
{
    this->target = target;
}

void DeepNetwork::printErrors()
{
    cout << "Total Error : " << this->error << endl;
}

void DeepNetwork::setCurrentInput(vector<double> input)
{
    this->input = input;

    for (int i = 0; i < input.size(); i++)
    {
        this->layers[0]->setVal(i, input[i]);
        this->layers[0]->getNeuron(i)->Activate();
        this->layers[0]->getNeuron(i)->Derive();
    }
}

void DeepNetwork::printToConsole()
{
    for (int i = 0; i < input.size(); i++)
    {
        cout << input.at(i) << "\t\t";
    }
    cout << endl;

    for (int i = 0; i < layers.at(layers.size() - 1)->getSize(); i++)
    {
        cout << layers.at(layers.size() - 1)->getNeurons().at(i)->getActivatedVal() << "\t";
    }
}

Layer* DeepNetwork::GetLayer(int nth)
{
    return layers[nth];
}

void DeepNetwork::printWeightMatrices()
{
    for (int i = 0; i < weightMatrices.size(); i++)
    {
        std::cout << "-------------------------------------------------------------" << endl;
        std::cout << "Weights for Hidden Layer : " << i + 1 << endl;
        weightMatrices[i]->printToConsole();
    }
}

void DeepNetwork::forwardPropogation()
{
    for (int i = 0; i < (int)layers.size() - 1; i++)
    {
        if (i + 1 < (int)layers.size()) {
            delete layers[i + 1];
            layers[i + 1] = nullptr;
        }
        layers[i + 1] = layers[i]->feedForward(
            weightMatrices[i], BaisMatrices[i],
            (i == 0),
            (i == (int)layers.size() - 2)
        );
    }
}
void DeepNetwork::printBiases()
{
    for (int i = 0; i < weightMatrices.size(); i++)
    {
        std::cout << "-------------------------------------------------------------" << endl;
        std::cout << "Bias for Hidden Layer : " << i + 1 << endl;
        BaisMatrices[i]->printToConsole();
    }
}

void DeepNetwork::setErrors()
{
    if (this->target.size() == 0)
    {
        cerr << "No target found" << endl;
        assert(false);
    }

    if (target.size() != layers[layers.size() - 1]->getNeurons().size())
    {
        cerr << "The size of the target is not equal to the size of the output" << endl;
        assert(false);
    }

    errors.resize(target.size());
    this->error = 0;
    int outputLayerIndx = this->layers.size() - 1;
    vector<Neuron*> outputNeurons = this->layers[outputLayerIndx]->getNeurons();

    for (int i = 0; i < target.size(); i++)
    {
        double act = target[i];
        double pred = outputNeurons[i]->getActivatedVal();

        double epsilon = 1e-15;
        pred = std::max(epsilon, std::min(1.0 - epsilon, pred));

        this->errors[i] = -act * std::log(pred);
        this->error += this->errors[i];
    }
}

double DeepNetwork::getGlobalError()
{
    return this->error;
}

void DeepNetwork::saveThisError(double error)
{
    this->histErrors.push_back(error);
}

void DeepNetwork::gardientComputation()
{
    int outputLayerIndex = this->topology.size() - 1;

    // Clear old gradients
    for (auto g : this->GradientMatrices) delete g;
    this->GradientMatrices.clear();

    // Snapshot all layer activations and derived values RIGHT NOW before
    // anything changes. Fixes the stale-read bug: feedForward() replaces
    // layer pointers each sample, so by the time updateWeights() runs
    // the live layer pointers may already point to the next sample's state.
    vector<GeneralMatrix::Matrix*> activationSnapshots;
    vector<GeneralMatrix::Matrix*> derivedSnapshots;
    for (int i = 0; i <= outputLayerIndex; i++) {
        activationSnapshots.push_back(this->layers.at(i)->convertTOMatrixActivatedVal());
        derivedSnapshots.push_back(this->layers.at(i)->convertTOMatrixDerivedVal());
    }

    // Output layer: softmax + cross-entropy combined derivative = pred - target
    GeneralMatrix::Matrix* outGrad = new GeneralMatrix::Matrix(1, this->topology.at(outputLayerIndex), false);
    for (int i = 0; i < this->topology.at(outputLayerIndex); i++) {
        double pred = this->layers.at(outputLayerIndex)->getNeuron(i)->getActivatedVal();
        outGrad->setVal(0, i, pred - this->target[i]);
    }
    this->GradientMatrices.push_back(outGrad);
    // GradientMatrices[0] = output gradient
    // GradientMatrices[1] = second-to-last layer gradient
    // ... (stored output -> input)

    // Backpropagate through hidden layers
    // weightMatrices[i] connects layer i to layer i+1
    for (int i = outputLayerIndex - 1; i > 0; i--)
    {
        GeneralMatrix::Matrix* lastGradient = this->GradientMatrices.back();
        GeneralMatrix::Matrix* Wt = this->weightMatrices.at(i)->tranpose();
        GeneralMatrix::Matrix* backpropGrad = *lastGradient * Wt;

        GeneralMatrix::Matrix* newGradients = new GeneralMatrix::Matrix(1, this->topology.at(i), false);
        for (int c = 0; c < this->topology.at(i); c++) {
            double g = backpropGrad->getVal(0, c) * derivedSnapshots[i]->getVal(0, c);
            newGradients->setVal(0, c, g);
        }
        this->GradientMatrices.push_back(newGradients);

        delete Wt;
        delete backpropGrad;
    }

    // Store snapshots so updateWeights() can use them safely
    for (auto m : this->layerActivationSnapshots) delete m;
    for (auto m : this->layerDerivedSnapshots) delete m;
    this->layerActivationSnapshots = activationSnapshots;
    this->layerDerivedSnapshots = derivedSnapshots;
}

std::vector<GeneralMatrix::Matrix*> DeepNetwork::GetWeightMatrices()
{
    return this->weightMatrices;
}

std::vector<GeneralMatrix::Matrix*> DeepNetwork::GetBiasMatrices()
{
    return this->BaisMatrices;
}

void DeepNetwork::updateWeights()
{
    // GradientMatrices layout (output -> input):
    //   [0] = output layer gradient       pairs with weightMatrices[numLayers-1]
    //   [1] = last hidden layer gradient  pairs with weightMatrices[numLayers-2]
    //   ...
    // weightMatrices layout (input -> output):
    //   [0] = layer0 -> layer1
    //   [1] = layer1 -> layer2
    //   ...
    // Fix: gradient[gradIdx] pairs with weightMatrices[numLayers-1-gradIdx]
    // No reversal needed — update in place.

    int numLayers = (int)this->weightMatrices.size();

    for (int gradIdx = 0; gradIdx < numLayers; gradIdx++)
    {
        int wIdx = numLayers - 1 - gradIdx;

        GeneralMatrix::Matrix* gradient = this->GradientMatrices[gradIdx];

        // Activations feeding into weightMatrices[wIdx] come from layer wIdx.
        // Use snapshots — live layers may already be from the next sample.
        GeneralMatrix::Matrix* prevAct;
        bool ownPrevAct = false;

        if (wIdx == 0) {
            // Input layer: use raw (pre-activation) values
            prevAct = this->layers.at(0)->convertTOMatrixVal();
            ownPrevAct = true;
        }
        else {
            prevAct = this->layerActivationSnapshots[wIdx];
            ownPrevAct = false;
        }

        GeneralMatrix::Matrix* prevT = prevAct->tranpose();
        GeneralMatrix::Matrix* deltaW = *prevT * gradient;

        // Update weights in place (no new matrix, no reversal)
        for (int r = 0; r < this->weightMatrices.at(wIdx)->getNumRow(); r++) {
            for (int c = 0; c < this->weightMatrices.at(wIdx)->getNumCols(); c++) {
                double orig = this->weightMatrices.at(wIdx)->getVal(r, c);
                double delta = deltaW->getVal(r, c);
                delta = std::max(-1.0, std::min(1.0, delta));
                this->weightMatrices.at(wIdx)->setVal(r, c, orig - this->learningRate * delta);
            }
        }

        // Update biases in place
        for (int c = 0; c < this->BaisMatrices.at(wIdx)->getNumCols(); c++) {
            double orig = this->BaisMatrices.at(wIdx)->getVal(0, c);
            double delta = gradient->getVal(0, c);
            delta = std::max(-1.0, std::min(1.0, delta));
            this->BaisMatrices.at(wIdx)->setVal(0, c, orig - this->learningRate * delta);
        }

        if (ownPrevAct) delete prevAct;
        delete prevT;
        delete deltaW;
    }
}

std::vector<GeneralMatrix::Matrix*> DeepNetwork::averageGradients(vector<vector<GeneralMatrix::Matrix*>> accgrad)
{
    size_t numColumns = accgrad[0].size();
    vector<GeneralMatrix::Matrix*> averageMatrices(numColumns, nullptr);

    for (size_t col = 0; col < numColumns; ++col)
    {
        averageMatrices[col] = new GeneralMatrix::Matrix(*accgrad[0][col]);
        for (int i = 0; i < averageMatrices[col]->getNumRow(); ++i)
            for (int j = 0; j < averageMatrices[col]->getNumCols(); ++j)
                averageMatrices[col]->setVal(i, j, 0.0);
    }

    for (int col = 0; col < (int)numColumns; col++)
        for (int i = 0; i < (int)accgrad.size(); i++)
            averageMatrices[col] = *averageMatrices[col] + accgrad[i][col];

    for (size_t col = 0; col < numColumns; ++col)
        averageMatrices[col]->divideByScalar(accgrad.size());

    this->GradientMatrices = averageMatrices;
    return averageMatrices;
}

std::vector<GeneralMatrix::Matrix*> DeepNetwork::GetGradientMatrices()
{
    return this->GradientMatrices;
}