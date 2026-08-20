#include "cnn/Layer.hpp"
#include <numeric>
#include <cmath>
#include <limits>

Layer::Layer(int size)
{
    this->size = size;
    for (int i = 0; i < size; i++)
    {
        Neuron* n = new Neuron(0.00);
        n->Activate();
        n->Derive();
        this->neurons.push_back(n);
    }
}

Layer::Layer(int size, bool last)
{
    this->size = size;
    for (int i = 0; i < size; i++)
    {
        Neuron* n = new Neuron(0.00);
        // Don't activate/derive yet - will be done in feedForward
        this->neurons.push_back(n);
    }
}

int Layer::getSize()
{
    return size;
}

GeneralMatrix::Matrix* Layer::convertTOMatrixVal()
{
    GeneralMatrix::Matrix* m = new GeneralMatrix::Matrix(1, this->neurons.size(), false);
    for (int i = 0; i < neurons.size(); i++)
    {
        m->setVal(0, i, this->neurons[i]->getVal());
    }
    return m;
}

GeneralMatrix::Matrix* Layer::convertTOMatrixActivatedVal()
{
    GeneralMatrix::Matrix* m = new GeneralMatrix::Matrix(1, this->neurons.size(), false);
    for (int i = 0; i < neurons.size(); i++)
    {
        m->setVal(0, i, this->neurons[i]->getActivatedVal());
    }
    return m;
}

GeneralMatrix::Matrix* Layer::convertTOMatrixDerivedVal()
{
    GeneralMatrix::Matrix* m = new GeneralMatrix::Matrix(1, this->neurons.size(), false);
    for (int i = 0; i < neurons.size(); i++)
    {
        m->setVal(0, i, this->neurons[i]->getDerivedVal());
    }
    return m;
}

Layer* Layer::feedForward(GeneralMatrix::Matrix* Weights, GeneralMatrix::Matrix* bias, bool isFirst, bool isLast)
{
    GeneralMatrix::Matrix* this_layer_val;
    if (isFirst)
    {
        this_layer_val = convertTOMatrixVal();
    }
    else {
        this_layer_val = convertTOMatrixActivatedVal();
    }

    GeneralMatrix::Matrix* z = *this_layer_val * Weights;
    GeneralMatrix::Matrix* zWithBias = *z + bias;

    if (!isLast) {
        // Hidden layer - apply activation function
        Layer* temp = new Layer(Weights->getNumCols());
        for (int i = 0; i < Weights->getNumCols(); i++)
        {
            temp->setVal(i, zWithBias->getVal(0, i));
            temp->getNeuron(i)->Activate();
            temp->getNeuron(i)->Derive();
        }

        delete this_layer_val;
        delete z;
        delete zWithBias;
        return temp;
    }
    else {
        // Output layer - apply softmax with improved numerical stability
        Layer* temp = new Layer(Weights->getNumCols(), true);

        // Set pre-activation values and find max for stability
        double maxVal = -std::numeric_limits<double>::infinity();
        for (int i = 0; i < Weights->getNumCols(); i++)
        {
            double val = zWithBias->getVal(0, i);
            // Clip extreme values to prevent overflow
            val = std::max(-500.0, std::min(500.0, val));
            temp->setVal(i, val);
            maxVal = std::max(maxVal, val);
        }

        // Compute softmax with numerical stability
        std::vector<double> expVals(temp->getNeurons().size());
        double sumExp = 0.0;

        for (size_t i = 0; i < temp->getNeurons().size(); i++) {
            double shiftedVal = temp->getNeurons().at(i)->getVal() - maxVal;
            expVals[i] = std::exp(shiftedVal);
            sumExp += expVals[i];
        }

        // Ensure sumExp is not too small
        if (sumExp < 1e-10) {
            sumExp = 1e-10;
        }

        // Set activated values (softmax outputs) with epsilon for numerical stability
        const double epsilon = 1e-7;
        for (size_t i = 0; i < temp->getNeurons().size(); i++) {
            double softmaxOutput = expVals[i] / sumExp;

            // Clip to prevent log(0) in cross-entropy
            softmaxOutput = std::max(epsilon, std::min(1.0 - epsilon, softmaxOutput));

            temp->setActivatedVal(i, softmaxOutput);

            // Set derivative (not used in softmax-crossentropy gradient, but kept for consistency)
            temp->getNeuron(i)->setDerivedVal(softmaxOutput * (1.0 - softmaxOutput));
        }

        delete this_layer_val;
        delete z;
        delete zWithBias;
        return temp;
    }
}

Neuron* Layer::getNeuron(int pos)
{
    return neurons.at(pos);
}

vector<Neuron*> Layer::getNeurons()
{
    return neurons;
}

void Layer::setVal(int i, double v)
{
    this->neurons[i]->setVal(v);
}

void Layer::setActivatedVal(int i, double v0)
{
    this->neurons[i]->setActivatedVal(v0);
}