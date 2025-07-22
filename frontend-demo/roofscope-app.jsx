import React, { useState, useCallback } from 'react';
import { Upload, Camera, Zap, Eye, Target, BarChart3, CheckCircle, AlertCircle } from 'lucide-react';

const RoofScopeApp = () => {
  const [selectedImage, setSelectedImage] = useState(null);
  const [imagePreview, setImagePreview] = useState(null);
  const [analysisResults, setAnalysisResults] = useState(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [error, setError] = useState(null);

  const handleImageUpload = useCallback((event) => {
    const file = event.target.files[0];
    if (file) {
      setSelectedImage(file);
      setError(null);
      setAnalysisResults(null);
      
      const reader = new FileReader();
      reader.onload = (e) => {
        setImagePreview(e.target.result);
      };
      reader.readAsDataURL(file);
    }
  }, []);

  const handleDrop = useCallback((event) => {
    event.preventDefault();
    const file = event.dataTransfer.files[0];
    if (file && file.type.startsWith('image/')) {
      const fakeEvent = { target: { files: [file] } };
      handleImageUpload(fakeEvent);
    }
  }, [handleImageUpload]);

  const handleDragOver = useCallback((event) => {
    event.preventDefault();
  }, []);

  const analyzeImage = async () => {
    if (!selectedImage) return;

    setIsAnalyzing(true);
    setError(null);

    try {
      const formData = new FormData();
      formData.append('image', selectedImage);

      const response = await fetch('/api/analyze', {
        method: 'POST',
        body: formData
      });

      const data = await response.json();

      if (data.status === 'success') {
        setAnalysisResults(data);
      } else {
        throw new Error(data.message || 'Analysis failed');
      }
    } catch (err) {
      setError(err.message);
      console.error('Analysis error:', err);
    } finally {
      setIsAnalyzing(false);
    }
  };

  const getQualityColor = (value, type) => {
    if (type === 'coverage') {
      return value > 20 ? 'text-green-600' : value > 10 ? 'text-yellow-600' : 'text-red-600';
    }
    if (type === 'obstacles') {
      return value > 5 ? 'text-red-600' : value > 2 ? 'text-yellow-600' : 'text-green-600';
    }
    return 'text-gray-600';
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 via-white to-green-50">
      {/* Header */}
      <div className="bg-white shadow-lg border-b">
        <div className="max-w-7xl mx-auto px-4 py-6">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-4">
              <div className="p-3 bg-blue-600 rounded-xl">
                <Camera className="w-8 h-8 text-white" />
              </div>
              <div>
                <h1 className="text-3xl font-bold text-gray-900">RoofScope</h1>
                <p className="text-gray-600">Automatic Roof Detection & Analysis using Deep Learning</p>
              </div>
            </div>
            
            <div className="text-right">
              <div className="text-sm text-gray-500">Performance Metrics</div>
              <div className="flex space-x-4 text-sm font-semibold">
                <span className="text-green-600">Accuracy: 97.65%</span>
                <span className="text-blue-600">IoU: 88.86%</span>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Main Content */}
      <div className="max-w-7xl mx-auto px-4 py-8">
        <div className="grid lg:grid-cols-2 gap-8">
          
          {/* Left Panel - Upload */}
          <div className="space-y-6">
            <div className="bg-white rounded-2xl shadow-xl border p-6">
              <h2 className="text-xl font-bold mb-4 flex items-center">
                <Upload className="w-5 h-5 mr-2 text-blue-600" />
                Upload Satellite Image
              </h2>
              
              <div 
                className={`border-2 border-dashed rounded-xl p-8 text-center transition-colors ${
                  imagePreview ? 'border-green-300 bg-green-50' : 'border-gray-300 hover:border-blue-400 hover:bg-blue-50'
                }`}
                onDrop={handleDrop}
                onDragOver={handleDragOver}
              >
                {imagePreview ? (
                  <div className="space-y-4">
                    <img 
                      src={imagePreview} 
                      alt="Preview" 
                      className="max-w-full h-48 object-contain mx-auto rounded-lg shadow-md"
                    />
                    <div className="text-green-600 font-medium">
                      <CheckCircle className="w-5 h-5 inline mr-2" />
                      Image uploaded successfully
                    </div>
                  </div>
                ) : (
                  <div className="space-y-4">
                    <Camera className="w-16 h-16 text-gray-400 mx-auto" />
                    <div>
                      <p className="text-lg text-gray-600">Drop satellite image here</p>
                      <p className="text-sm text-gray-500">or click to browse</p>
                    </div>
                  </div>
                )}
                
                <input
                  type="file"
                  accept="image/*"
                  onChange={handleImageUpload}
                  className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
                />
              </div>
              
              <button
                onClick={analyzeImage}
                disabled={!selectedImage || isAnalyzing}
                className={`w-full mt-4 px-6 py-3 rounded-xl font-semibold transition-all ${
                  selectedImage && !isAnalyzing
                    ? 'bg-blue-600 text-white hover:bg-blue-700 transform hover:scale-105 shadow-lg'
                    : 'bg-gray-300 text-gray-500 cursor-not-allowed'
                }`}
              >
                {isAnalyzing ? (
                  <div className="flex items-center justify-center">
                    <div className="animate-spin rounded-full h-5 w-5 border-2 border-white border-t-transparent mr-2"></div>
                    Analyzing Image...
                  </div>
                ) : (
                  <div className="flex items-center justify-center">
                    <Zap className="w-5 h-5 mr-2" />
                    Analyze Roof
                  </div>
                )}
              </button>
            </div>

            {/* Features */}
            <div className="bg-white rounded-2xl shadow-xl border p-6">
              <h3 className="text-lg font-bold mb-4">Analysis Features</h3>
              <div className="space-y-3">
                <div className="flex items-center space-x-3">
                  <Eye className="w-5 h-5 text-green-600" />
                  <span className="text-gray-700">U-Net Semantic Segmentation</span>
                </div>
                <div className="flex items-center space-x-3">
                  <Target className="w-5 h-5 text-blue-600" />
                  <span className="text-gray-700">Canny Edge Detection</span>
                </div>
                <div className="flex items-center space-x-3">
                  <AlertCircle className="w-5 h-5 text-orange-600" />
                  <span className="text-gray-700">MSER Obstacle Detection</span>
                </div>
                <div className="flex items-center space-x-3">
                  <BarChart3 className="w-5 h-5 text-purple-600" />
                  <span className="text-gray-700">Comprehensive Metrics</span>
                </div>
              </div>
            </div>
          </div>

          {/* Right Panel - Results */}
          <div className="space-y-6">
            {error && (
              <div className="bg-red-50 border border-red-200 rounded-2xl p-6">
                <div className="flex items-center text-red-800">
                  <AlertCircle className="w-5 h-5 mr-2" />
                  <strong>Analysis Error</strong>
                </div>
                <p className="text-red-700 mt-2">{error}</p>
              </div>
            )}

            {analysisResults ? (
              <div className="space-y-6">
                {/* Visualization */}
                <div className="bg-white rounded-2xl shadow-xl border p-6">
                  <h3 className="text-xl font-bold mb-4">Analysis Results</h3>
                  <img 
                    src={analysisResults.results.visualization}
                    alt="Analysis Results" 
                    className="w-full rounded-lg shadow-md"
                  />
                </div>

                {/* Metrics */}
                <div className="bg-white rounded-2xl shadow-xl border p-6">
                  <h3 className="text-lg font-bold mb-4">Quantitative Metrics</h3>
                  <div className="grid grid-cols-2 gap-4">
                    <div className="bg-blue-50 rounded-xl p-4">
                      <div className="text-2xl font-bold text-blue-600">
                        {analysisResults.results.roof_coverage_percent}%
                      </div>
                      <div className="text-sm text-gray-600">Roof Coverage</div>
                    </div>
                    
                    <div className="bg-orange-50 rounded-xl p-4">
                      <div className="text-2xl font-bold text-orange-600">
                        {analysisResults.results.obstacles_detected}
                      </div>
                      <div className="text-sm text-gray-600">Obstacles Found</div>
                    </div>
                    
                    <div className="bg-green-50 rounded-xl p-4">
                      <div className="text-2xl font-bold text-green-600">
                        {analysisResults.results.edge_contours_found}
                      </div>
                      <div className="text-sm text-gray-600">Edge Contours</div>
                    </div>
                    
                    <div className="bg-purple-50 rounded-xl p-4">
                      <div className="text-sm font-bold text-purple-600">
                        {analysisResults.results.image_dimensions}
                      </div>
                      <div className="text-sm text-gray-600">Resolution</div>
                    </div>
                  </div>
                </div>

                {/* Analysis Summary */}
                <div className="bg-white rounded-2xl shadow-xl border p-6">
                  <h3 className="text-lg font-bold mb-4">Analysis Summary</h3>
                  <div className="space-y-3">
                    <div className="flex justify-between items-center">
                      <span className="text-gray-600">Segmentation Quality:</span>
                      <span className={`font-semibold ${
                        analysisResults.analysis.segmentation_quality === 'High' 
                          ? 'text-green-600' 
                          : 'text-yellow-600'
                      }`}>
                        {analysisResults.analysis.segmentation_quality}
                      </span>
                    </div>
                    
                    <div className="flex justify-between items-center">
                      <span className="text-gray-600">Obstacle Density:</span>
                      <span className={`font-semibold ${
                        analysisResults.analysis.obstacle_density === 'High' 
                          ? 'text-red-600' 
                          : 'text-green-600'
                      }`}>
                        {analysisResults.analysis.obstacle_density}
                      </span>
                    </div>
                    
                    <div className="flex justify-between items-center">
                      <span className="text-gray-600">Edge Complexity:</span>
                      <span className={`font-semibold ${
                        analysisResults.analysis.edge_complexity === 'Complex' 
                          ? 'text-orange-600' 
                          : 'text-blue-600'
                      }`}>
                        {analysisResults.analysis.edge_complexity}
                      </span>
                    </div>
                  </div>
                </div>

                {/* Applications */}
                <div className="bg-gradient-to-r from-blue-50 to-green-50 rounded-2xl border p-6">
                  <h3 className="text-lg font-bold mb-3">Recommended Applications</h3>
                  <div className="text-sm space-y-2">
                    {analysisResults.results.roof_coverage_percent > 15 && (
                      <div className="flex items-center text-green-700">
                        <CheckCircle className="w-4 h-4 mr-2" />
                        Suitable for solar panel installation
                      </div>
                    )}
                    {analysisResults.results.obstacles_detected <= 3 && (
                      <div className="flex items-center text-blue-700">
                        <CheckCircle className="w-4 h-4 mr-2" />
                        Low obstacle density - good for maintenance access
                      </div>
                    )}
                    {analysisResults.analysis.edge_complexity === 'Simple' && (
                      <div className="flex items-center text-purple-700">
                        <CheckCircle className="w-4 h-4 mr-2" />
                        Simple geometry - easier installation planning
                      </div>
                    )}
                  </div>
                </div>
              </div>
            ) : (
              <div className="bg-white rounded-2xl shadow-xl border p-8 text-center">
                <Camera className="w-16 h-16 text-gray-300 mx-auto mb-4" />
                <p className="text-gray-500 text-lg">Upload a satellite image to see detailed roof analysis</p>
                <p className="text-gray-400 text-sm mt-2">
                  Our AI will detect roofs, analyze edges, and identify obstacles like solar panels or chimneys
                </p>
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Footer */}
      <div className="bg-gray-50 border-t mt-12">
        <div className="max-w-7xl mx-auto px-4 py-8">
          <div className="text-center space-y-4">
            <h3 className="text-lg font-semibold text-gray-800">
              RoofScope: Advanced Computer Vision for Roof Analysis
            </h3>
            <div className="grid md:grid-cols-3 gap-6 text-sm text-gray-600 max-w-4xl mx-auto">
              <div>
                <strong className="text-gray-800">Deep Learning:</strong>
                <br />U-Net architecture for precise semantic segmentation of roof structures from satellite imagery
              </div>
              <div>
                <strong className="text-gray-800">Computer Vision:</strong>
                <br />Canny edge detection and MSER algorithms for structural analysis and obstacle identification
              </div>
              <div>
                <strong className="text-gray-800">Applications:</strong>
                <br />Solar panel planning, roof maintenance, urban development, and architectural analysis
              </div>
            </div>
            <div className="pt-4 border-t border-gray-200">
              <p className="text-gray-500">
                Developed for CS484: Introduction to Computer Vision | 
                <span className="font-semibold"> Performance: 97.65% Accuracy, 88.86% IoU</span>
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default RoofScopeApp;