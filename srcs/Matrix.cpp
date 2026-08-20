#include "cnn/Matrix.hpp"
#include "cnn/Random.hpp"

gridEntity CNN_Matrix::Matrix::convolute(gridEntity input_image_section, gridEntity filter, int stride)
{
    int inputHeight = input_image_section.size();
    int inputWidth = input_image_section[0].size();
    int filterHeight = filter.size();
    int filterWidth = filter[0].size();

    // Check if input and filter dimensions are valid
    if (inputHeight < filterHeight || inputWidth < filterWidth) {
        throw std::invalid_argument("Filter dimensions must be smaller than or equal to input dimensions.");
    }

    // Calculate dimensions of the output feature map
    int outputHeight = inputHeight - filterHeight + 1;
    int outputWidth = inputWidth - filterWidth + 1;

    // Initialize the output grid
    gridEntity output(outputHeight, std::vector<double>(outputWidth, 0.0));

    // Perform convolution
    for (int i = 0; i < outputHeight; ++i) {
        for (int j = 0; j < outputWidth; ++j) {
            double sum = 0.0;

            // Apply filter to the current receptive field
            for (int m = 0; m < filterHeight; ++m) {
                for (int n = 0; n < filterWidth; ++n) {
                    sum += input_image_section[i + m][j + n] * filter[m][n];
                }
            }

            // Store result in the output grid


            // I can probaly just apply RELU Here !?????
            output[i][j] = sum;
        }
    }

    return output;
}

double CNN_Matrix::Matrix::genRandomNumber()
{
    // Draws from the single shared engine (Random::engine()) instead of
    // constructing a fresh random_device-seeded mt19937 on every call.
    // This is what makes matrix randomization reproducible across runs
    // when Random::seed(...) is set once at program start.
    std::uniform_real_distribution<float> dis(-3, 3);
    return dis(Random::engine());
}

void CNN_Matrix::Matrix::randomize_all_values(gridEntity& mat, int numRows, int numCols)
{
    gridEntity temp;
    for (int i = 0; i < numRows; i++)
    {
        std::vector<double> cols;
        for (int j = 0; j < numCols; j++)
        {
            double r = 0.00;
            r = CNN_Matrix::Matrix::genRandomNumber();
            cols.push_back(r);
        }
        temp.push_back(cols);
    }
    mat = temp;
}


double CNN_Matrix::Matrix::sum_of_all_elements(gridEntity matrix)
{
    double sum = 0.0;
    for (const auto& row : matrix) {
        for (double elem : row) {
            sum += elem;
        }
    }
    return sum;
}


gridEntity CNN_Matrix::Matrix::sum_of_all_matrix_elements(std::vector<gridEntity> all_matrices)
{
    int rows = all_matrices[0].size();
    int cols = all_matrices[0][0].size();

    gridEntity result(rows, std::vector<double>(cols, 0));

    for (const auto& mat : all_matrices) {
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                result[i][j] += mat[i][j];
            }
        }
    }
    return result;
}

void CNN_Matrix::Matrix::displayMatrix(gridEntity mat) {
    for (auto rows : mat)
    {
        for (auto vals : rows)
        {
            std::cout << vals << " ";
        }
        std::cout << "\n";
    }
}



GeneralMatrix::Matrix::Matrix(int numRows, int numCols, bool isRandom = true)
{
    this->numRows = numRows;
    this->numCols = numCols;

    for (int i = 0; i < numRows; i++)
    {
        std::vector<double> cols;
        for (int j = 0; j < numCols; j++)
        {
            double r = 0.00;
            if (isRandom)
                r = this->genRandomNumber();
            cols.push_back(r);
        }
        this->values.push_back(cols);
    }
}


double GeneralMatrix::Matrix::genRandomNumber()
{
    // He initialization: N(0, 0.1)
    // MUST include negative values or all hidden activations are positive,
    // output logits are all identical, softmax is uniform, loss = ln(10) forever.
    //
    // Draws from the single shared engine (Random::engine()) instead of a
    // fresh random_device-seeded mt19937 per call, so weight init is
    // reproducible across runs when Random::seed(...) is set once at
    // program start.
    std::normal_distribution<double> dis(0.0, 0.1);
    return dis(Random::engine());
}

void GeneralMatrix::Matrix::printToConsole()
{
    for (int i = 0; i < numRows; i++)
    {
        for (int j = 0; j < numCols; j++)
        {
            std::cout << " " << this->values[i][j] << "\t";
        }
        std::cout << std::endl;
    }
}

void GeneralMatrix::Matrix::divideByScalar(double scalar) {
    for (int i = 0; i < numRows; ++i) {
        for (int j = 0; j < numCols; ++j) {
            values[i][j] /= scalar;
        }
    }
}
void GeneralMatrix::Matrix::setVal(int r, int c, double val)
{
    this->values[r][c] = val;
}

double GeneralMatrix::Matrix::getVal(int r, int c)
{
    return this->values[r][c];
}

GeneralMatrix::Matrix* GeneralMatrix::Matrix::tranpose()
{
    Matrix* tans = new Matrix(this->numCols, this->numRows, false);
    for (int orgRow = 0; orgRow < this->numRows; orgRow++)
    {
        for (int orgCol = 0; orgCol < this->numCols; orgCol++)
        {
            tans->values[orgCol][orgRow] = getVal(orgRow, orgCol);
        }
    }
    return tans;
}
GeneralMatrix::Matrix* GeneralMatrix::Matrix::operator *(Matrix*& B)
{
    int rows_A = numRows;
    int cols_A = numCols;
    int cols_B = B->getNumCols();
    int rows_B = B->getNumRow();
    // A vaneko Aafu, B vaneko arko
    // Resultant matrix C with size rows_A x
    Matrix* C = new Matrix(rows_A, cols_B, false);

    if (cols_A != rows_B)
    {
        std::cout << "-------------------------------------------------------------" << std::endl;
        std::cout << "Invlaid Dimentions" << std::endl;
        std::cout << std::endl;
        std::cout << std::endl;
        std::cout << " This Matrix " << std::endl;
        printToConsole();

        std::cout << " Passed Matrix " << std::endl;
        B->printToConsole();
        std::cout << std::endl;
        std::cout << std::endl;

        return C;
    }

    for (int i = 0; i < rows_A; ++i)
    {
        for (int j = 0; j < cols_B; ++j)
        {
            for (int k = 0; k < cols_A; ++k)
            {
                C->values[i][j] += values[i][k] * B->values[k][j];
            }
        }
    }

    return C;
}

GeneralMatrix::Matrix* GeneralMatrix::Matrix::operator +(Matrix*& B)
{
    int rows = B->getNumRow();
    int cols = B->getNumCols(); // Assuming both matrices have the same dimensions

    Matrix* ans = new Matrix(rows, cols, false);

    for (int i = 0; i < rows; ++i)
    {
        for (int j = 0; j < cols; ++j)
        {
            ans->values[i][j] += values[i][j] + B->values[i][j];
        }
    }

    return ans;
}