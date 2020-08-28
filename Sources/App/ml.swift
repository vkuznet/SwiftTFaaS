// load python module
#if canImport(PythonKit)
    import PythonKit
#else
    import Python
#endif

import TensorFlow

var model = TFModel()
func loadModel() {
    model.loadWeights(numpyFile: "model/model.tf")
}

// TFModel provides TensorFlow model we'll use for training
// the model has three layers
// (4, 10, relu) -> (10, 10, relu) -> (10, 3)
struct TFModel: Layer {
    var layer1 = Dense<Float>(inputSize: 4, outputSize: 10, activation: relu)
    var layer2 = Dense<Float>(inputSize: 10, outputSize: 10, activation: relu)
    var layer3 = Dense<Float>(inputSize: 10, outputSize: 3)

    @differentiable
    func callAsFunction(_ input: Tensor<Float>) -> Tensor<Float> {
        return input.sequenced(through: layer1, layer2, layer3)
    }
}

// saving layer protocol weights, see
// https://gist.github.com/kongzii/62b9d978a6536bb97095ed3fb74e30fd
// later we should switch to Checkpoints Reader/Writer, see
// https://github.com/tensorflow/swift-models/tree/master/Checkpoints
extension Layer {
    mutating public func loadWeights(numpyFile: String) {
        print("loading weights from: \(numpyFile).npy")
        let np = Python.import("numpy")
        let weights = np.load(numpyFile+".npy", allow_pickle: true)

        for (index, kp) in self.recursivelyAllWritableKeyPaths(to:  Tensor<Float>.self).enumerated() {
            self[keyPath: kp] = Tensor<Float>(numpy: weights[index])!
        }
    }
}

// helper function to print our predictions
func printPredictions(classNames: [String], preds: Tensor<Float>) {
    for i in 0..<preds.shape[0] {
        let logits = preds[i]
        let classIdx = logits.argmax().scalar!
        print("Example \(i) prediction: \(classNames[Int(classIdx)]) (\(softmax(logits)))")
    }
}
func outputPredictions(classNames: [String], preds: Tensor<Float>) -> [Float] {
    var array = [Float]()
    for i in 0..<preds.shape[0] {
        let logits = preds[i]
        let arr = softmax(logits).makeNumpyArray().tolist()
        for v in arr {
            let a: Float = Float(v)!
            array.append(a)
        }
    }
    return array
}
// helper function to provide unlabeled data for testing ML model
func unlabeledData() -> Tensor<Float> {
    let dataset: Tensor<Float> =
        [[5.1, 3.3, 1.7, 0.5],
         [5.9, 3.0, 4.2, 1.5],
         [6.9, 3.1, 5.4, 2.1]]
    return dataset
}
