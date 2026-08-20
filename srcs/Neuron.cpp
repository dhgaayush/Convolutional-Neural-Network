#include "cnn/Neuron.h"
#include <math.h>
using namespace std;

Neuron::Neuron(double val)
{
    this->val = val;
}

void Neuron::setVal(double val)
{
    this->val = val;
}

void Neuron::Activate()
{
    // Leaky ReLU activation
    this->activatedVal = this->val > 0 ? this->val : 0.01 * this->val;
}

void Neuron::Derive()
{
    // Derivative of Leaky ReLU
    this->derivedVal = this->val > 0 ? 1.0 : 0.01;
}

void Neuron::ActivateFinal()
{
    // For output layer - softmax will be applied externally
    // Just store the raw value for now
    this->activatedVal = this->val;
}

void Neuron::DeriveFinal()
{
    // For softmax + cross-entropy, the derivative simplifies to (y_pred - y_true)
    // This will be set externally during backprop
    // Initialize to 1.0 as a placeholder
    this->derivedVal = 1.0;
}

void Neuron::setActivatedVal(double val)
{
    this->activatedVal = val;
}

void Neuron::setDerivedVal(double val)
{
    this->derivedVal = val;
}

double Neuron::getVal()
{
    return this->val;
}

double Neuron::getActivatedVal()
{
    return this->activatedVal;
}

double Neuron::getDerivedVal()
{
    return this->derivedVal;
}