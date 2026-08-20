#pragma once


#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>
#include<string>
#include <vector>
#include <stdexcept>

using gridEntity = std::vector<std::vector<double>>;
using volumetricEntity = std::vector<std::vector<std::vector<double>>>;



namespace Filters{

	// NOTE: changed from `static` to `inline` (C++17).
	// `static` at namespace scope gives internal linkage, meaning every
	// .cpp file that includes this header gets its OWN independent copy
	// of these matrices. `inline` guarantees a single shared definition
	// across all translation units instead. Since these values are
	// currently constant literals this wasn't causing incorrect output,
	// but it's a landmine for future changes (e.g. anything that ever
	// mutates a filter in place would silently diverge between files).
	inline gridEntity STRONG_VERTICAL_EDGE_DETECTION = {
		{ 1, 0, -1, 0, 1 },
		{ 1, 0, -1, 0, 1 },
		{ 1, 0, -1, 0, 1 },
		{ 1, 0, -1, 0, 1 },
		{ 1, 0, -1, 0, 1 } };
	inline gridEntity STRONG_HORIZONTAL_EDGE_DETECTION = {
		{1, 1, 1, 1, 1},
		{0, 0, 0, 0, 0},
		{-1, -1, -1, -1, -1},
		{0, 0, 0, 0, 0},
		{1, 1, 1, 1, 1} };
	inline gridEntity STRONG_DIAGONAL_EDGE_DETECTION = {
		{1, 1, 0, -1, -1},
		{1, 1, 0, -1, -1},
		{0, 0, 0, 0, 0},
		{-1, -1, 0, 1, 1},
		{-1, -1, 0, 1, 1} };
}
