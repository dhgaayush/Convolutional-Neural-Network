#pragma once
#include "cnn/ConvolutionLayers.hpp"



class ConvolutionLayers
{
	// The raw image we get from input layer
	gridEntity raw_image;

	// All the filters we need to apply to the main image
	// I can extract the no of filters, filter dimentions from this data
	// Do note that, the filers are 2d filters as grid entity is a (2D) Matrix
	std::vector<gridEntity> predefined_filters;

	// The results after appying the filters to the main image
	// no. of filters = no. of feature_map
	std::vector<gridEntity> feature_maps;


	// Pool maps or smth, idk what they are called
	std::vector<gridEntity> pool_maps;


	// Define Filters for 1'st Convolutoin layer [Deterministic] -> Still there are better places to put filter


	// Define Filters for 2'nd Convolution layer [Non-Determinstic/The shit to train]
	std::vector<gridEntity> trained_filters;
	// Assume i need to apply 4 filters( 3 x 3 x P) , P because i applied P filters in the last convolution layer to get P feature maps so it became P chanal input for my new convolution layer
	int no_of_filters_in_second_CL;
	volumetricEntity input_channels;
	std::vector<volumetricEntity> training_filters; // each kernal will have a dimention of say 3x3xN where N is the no of pooled maps 
	volumetricEntity ouput_feature_maps;

	
	std::vector<gridEntity> final_pool_maps;


	// Informations
	int no_of_filters_used; // = no_of_channels

public:
	ConvolutionLayers(gridEntity);


	// std::vector<gridEntity> getFeatureMaps();
	// std::vector<gridEntity> getPoolMaps();

	// Returns all the filters used int the first convolution layer
	std::vector<gridEntity> get_all_predefined_filter();

	// Returns all the training filters ijn the second convolution layer
	std::vector<volumetricEntity>& get_all_training_filter();

	// Returns the raw image in form fo gridEntity
	gridEntity &get_raw_input_image();

	// Return by reference as we may need to Insert into feature map
	std::vector<gridEntity>& get_feature_map();


	// Return by reference as we may need to Insert into input channel
	volumetricEntity& get_input_channels();

	// Return by reference as we may need to Insert into output features
	volumetricEntity& get_output_feature_maps();

	std::vector<gridEntity> &get_final_pool_maps();




	// Return by reference as we may need to Insert into pool map
	std::vector<gridEntity>& get_pool_map();

	// Takes image to apply filter to, the filter to apply and stride and returns the feature_map
	gridEntity apply_filter_universal(gridEntity, gridEntity, int);

	// Takes the reference to the feature map and activates it
	void activate_feature_map_using_RELU_universal(gridEntity&);
	void activate_feature_map_using_SIGMOID(gridEntity&);

	// Takes the reference to the feature map and applies normaization to the feature map
	void apply_normalaization_universal(gridEntity&);
	void set_raw_input_image(const gridEntity& image);

	// Takes a feature map and returns the pool map
	gridEntity apply_pooling_univeral(gridEntity, int);


	gridEntity unpool_without_indices( gridEntity& ,  gridEntity& , int , int , int );


	//std::vector<volumetricEntity> compute_filter_gradients(
	//	const std::vector<gridEntity>& inputChannels,        // Input to the convolutional layer
	//	const std::vector<gridEntity>& outputGradients,      // Gradients w.r.t output (from unpooling)
	//	int stride
	//);


	gridEntity cross_correlate(
		const gridEntity& input,
		const gridEntity& gradient,
		int stride
	);

	gridEntity full_convolve(
		const gridEntity& input,
		const gridEntity& filter,
		int stride
	);

	void apply_relu_derivative(
		gridEntity& gradients,
		const gridEntity& original_output
	);

	std::vector<gridEntity> compute_input_gradients(
		const std::vector<gridEntity>& outputGradients,
		const std::vector<volumetricEntity>& filters,
		int stride
	);

	// Update existing declarations if needed
	std::vector<volumetricEntity> compute_filter_gradients(
		const std::vector<gridEntity>& inputChannels,
		const std::vector<gridEntity>& outputGradients,
		int stride
	);

	void update_filters_with_gradients(
		std::vector<volumetricEntity>& filters,
		const std::vector<volumetricEntity>& gradients,
		double learningRate
	);

};

