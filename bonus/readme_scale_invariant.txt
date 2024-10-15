Custom Feature Descriptor for Scale Invariance : SIFT

To achieve scale invariance, our team implemented a simple SIFT descriptor.
This not only achieves scale invariance but also much higer performance in feature descriptions.


<<<Basic steps>>>

01 Preprocess
We converted to a floating-point data type and normalized, as the basic feature descriptor did.
Implemented this step to maintain the consistency in feature computation across images with different brightness.

02 Grayscale Conversion
Then we converted the image to graysclae, disregarding the color channel to focus on the intensity gradients.

03 Gradient Calculation
By employing the Sobel operator, we determined the gradients along the x and y axes, essential for revealing the orientation and edge directions within the image.


<<<SIFT implementation>>>

04 Keypoint Extraction
Feeded the keypoint value detected in the image.

05 SIFT Descriptor Construction
Based on the lecture, we constructed a 16x16 window around the keypoint.
We divided the window into 16 smaller, 4x4 sub regions.
Inside each sub-region, we created an 8 bin orientation histogram from the maginitudes and orientations of the gradients.
We also weighted the magnitude by a Gaussian window, giving more importance to gradients closer to the center keypoint.
Concatenating these histograms from all 16 regions formed a 128-element feature vector for each keypoint.

06 Normalization
Normalized the feature vector to unit length. In our insight, this process was may have been
useful for dealing with varying light conditions.


<<<Result>>>
By achieving not only scale but also rotation invariance (to some degree), 
the custom descriptor (Avg AUC: 0.9869) performed far better than the MOPS (Avg AUC: 0.9038) or simple descriptor (Avg AUC: 0.8947). 

This is due to the nature of MOPS being non-scale invariance unless it uses multi-scale detection, performing lower than SIFT.
